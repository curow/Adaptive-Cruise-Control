# ==============================================================================
# -- necessary module import -------------------------------
# ==============================================================================
import sys
import os
import glob
import random
import time
import math
import weakref
import numpy as np
import pandas as pd

# ==============================================================================
# -- find carla and other API module and add them to path ----------------------
# ==============================================================================
try:
    python_api_path = '/home/hollow/carla/CARLA_0.9.7/PythonAPI/carla/'
    carla_path = python_api_path + 'dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64')
    sys.path.append(glob.glob(carla_path)[0])
    sys.path.append(python_api_path)
except IndexError:
    print("Could't find carla module")

import carla
from agents.navigation.controller import VehiclePIDController

# ==============================================================================
# -- import helper function from local module --------------------------------------------
# ==============================================================================
from utils import main

# ==============================================================================
# --global constants  ---------------------------------------------------------------------------------------------------
# ==============================================================================
TIME_INTERVAL = 0.08

# ==============================================================================
# -- utility function ----------------------------------------------------------
# ==============================================================================
def draw_waypoints(world, waypoints, z=0.1):
    """
    Draw a list of waypoints at a certain height given in z.

    :param world: carla.world object
    :param waypoints: list or iterable container with the waypoints to draw
    :param z: height in meters
    :return:
    """
    for w in waypoints:
        t = w.transform
        begin = t.location + carla.Location(z=z)
        angle = math.radians(t.rotation.yaw)
        end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
        # world.debug.draw_arrow(begin, end, arrow_size=0.3)
        world.debug.draw_line(begin, end)

def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

def get_speed(vehicle, unit='kmh'):
    """
    Compute speed of a vehicle in Kmh
    :param vehicle: the vehicle for which speed is calculated
    :return: speed as a float in Kmh
    """
    vel = vehicle.get_velocity()
    vel_ms = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2) 
    if unit == 'kmh':
        return 3.6 * vel_ms
    else:
        return vel_ms

def emergency_stop():
        return carla.VehicleControl(
            steer=0,
            throttle=0,
            brake=1.0,
            hand_brake=False,
            manual_gear_shift=False
        )

# ==============================================================================
# -- Vehicle Physical Constrains -----------------------------------------------
# ==============================================================================
# https://github.com/carla-simulator/ros-bridge/blob/master/carla_ackermann_control/src/carla_ackermann_control/carla_control_physics.py

def get_vehicle_max_speed(_):
    """
    Get the maximum speed of a carla vehicle
    :param vehicle_info: the vehicle info
    :type vehicle_info: carla_ros_bridge.CarlaEgoVehicleInfo
    :return: maximum speed [m/s]
    :rtype: float64
    """
    # 180 km/h is the default max speed of a car
    max_speed = 180.0 / 3.6

    return max_speed


def get_vehicle_max_acceleration(_):
    """
    Get the maximum acceleration of a carla vehicle
    default: 3.0 m/s^2: 0-100 km/h in 9.2 seconds
    :param vehicle_info: the vehicle info
    :type vehicle_info: carla_ros_bridge.CarlaEgoVehicleInfo
    :return: maximum acceleration [m/s^2 > 0]
    :rtype: float64
    """
    max_acceleration = 6.0

    return max_acceleration


def get_vehicle_max_deceleration(_):
    """
    Get the maximum deceleration of a carla vehicle
    default: 8 m/s^2
    :param vehicle_info: the vehicle info
    :type vehicle_info: carla_ros_bridge.CarlaEgoVehicleInfo
    :return: maximum deceleration [m/s^2 > 0]
    :rtype: float64
    """
    max_deceleration = 8.0

    return max_deceleration

# ==============================================================================
# -- Intelligent Driver Model --------------------------------------------------
# ==============================================================================
class IDM:
    def __init__(self, vehicle, desired_speed, dt):
        """
        leader_speed: [kmh]
        """
        self.ego_vehicle = vehicle
        self.time_interval = dt
        self.desired_speed = desired_speed / 3.6
        # 2 second time headway
        self.safetime_headway = 2
        # 10 meters minimum distance
        self.minimum_distance = 10
        # leader information
        self.leader_distance = float('inf')
        self.leader_speed = float('inf')

        self.max_acceleration = get_vehicle_max_acceleration(self.ego_vehicle)
        self.max_deceleration = get_vehicle_max_deceleration(self.ego_vehicle)
    
    def set_leader_info(self, leader_distance, leader_speed):
        # translate kmh to meters/second
        self.leader_distance = leader_distance
        self.leader_speed = leader_speed / 3.6
    
    def get_leader_info(self):
        return self.leader_distance, self.leader_speed

    def calc_desired_gap(self):
        ego_speed = get_speed(self.ego_vehicle, unit='ms')
        additional_gap = (self.safetime_headway * ego_speed) + \
                (ego_speed * (ego_speed - self.leader_speed) /
                        (2 * math.sqrt(self.max_acceleration * self.max_deceleration)))
        return self.minimum_distance + max(0, additional_gap)

    def calc_desired_acceleration(self):
        """
        dv(t)/dt = a[1 - (v(t)/v0)^4  - (s*(t)/s(t))^2]
        """
        ego_speed = get_speed(self.ego_vehicle, unit='ms')
        free_ratio = math.pow(
                ego_speed / self.desired_speed, 4)
        block_ratio = math.pow(
                self.calc_desired_gap() / self.leader_distance, 2)
        return self.max_acceleration * (1 - free_ratio - block_ratio)

    def calc_desired_speed(self):
        dt = self.time_interval
        desired_acceleration = self.calc_desired_acceleration()
        ego_speed = get_speed(self.ego_vehicle, unit='ms')
        desired_speed = ego_speed + desired_acceleration * dt
        return max(desired_speed, 0) *3.6

# ==============================================================================
# -- Naive Agent ---------------------------------------------------------------
# ==============================================================================
class NaiveAgent:
    def __init__(self, vehicle, route, target_speed=20):
        self._vehicle = vehicle
        self._route = route
        self._target_speed = target_speed
        self._driver = IDM(self._vehicle, self._target_speed, TIME_INTERVAL)
        self._world = self._vehicle.get_world()
        self._map = self._world.get_map()
        self._vehicle_controller = VehiclePIDController(self._vehicle)

        self._history = pd.DataFrame(columns=[
            'timestamp'
            'ego_vehicle_x',
            'ego_vehicle_y',
            'ego_vehicle_z',
            'ego_vehicle_v',
            'leader_vehicle_x',
            'leader_vehicle_y',
            'leader_vehicle_z',
            'leader_vehicle_v',
        ])
    
    def update_history(self, info_dict):
        self._history = self._history.append(info_dict, ignore_index=True)

    @staticmethod
    def get_info_dict(timestamp, ego_vehicle_snapshot, leader_vehicle_snapshot):
        info_dict = {}
        info_dict['timestamp'] = timestamp.elapsed_seconds
        ego_vehicle_location = ego_vehicle_snapshot.get_transform().location
        info_dict['ego_vehicle_x'] = ego_vehicle_location.x
        info_dict['ego_vehicle_y'] = ego_vehicle_location.y
        info_dict['ego_vehicle_z'] = ego_vehicle_location.z
        info_dict['ego_vehicle_v'] = get_speed(ego_vehicle_snapshot)
        if leader_vehicle_snapshot:
            leader_vehicle_location = leader_vehicle_snapshot.get_transform().location
            info_dict['leader_vehicle_x'] = leader_vehicle_location.x
            info_dict['leader_vehicle_y'] = leader_vehicle_location.y
            info_dict['leader_vehicle_z'] = leader_vehicle_location.z
            info_dict['leader_vehicle_v'] = get_speed(leader_vehicle_snapshot)
        return info_dict

    def store_history(self):
        self._history.to_csv('./out/acc.csv')

    def set_target_speed(self, target_speed):
        self._target_speed = target_speed

    def is_close(self, other_transform, epsilon=0.1):
        vehicle_transform = self._vehicle.get_transform()
        loc = vehicle_transform.location
        dx = other_transform.location.x - loc.x
        dy = other_transform.location.y - loc.y

        return np.linalg.norm(np.array([dx, dy])) < epsilon
    
    def is_ahead(self, other_transform):
        """Check if ego vehicle is ahead of other"""
        vehicle_transform = self._vehicle.get_transform()
        loc = vehicle_transform.location
        orientation = vehicle_transform.rotation.yaw

        dx = other_transform.location.x - loc.x
        dy = other_transform.location.y - loc.y
        target_vector = np.array([dx, dy])
        norm_target = np.linalg.norm(target_vector)

        forward_vector = np.array([
            math.cos(math.radians(orientation)),
            math.sin(math.radians(orientation))
            ])
        cos_vector = np.dot(forward_vector, target_vector) / norm_target
        if cos_vector > 1:
            cos_vector = 1
        elif cos_vector < -1:
            cos_vector = -1
        d_angle = math.degrees(math.acos(cos_vector))

        return d_angle > 90

    def get_distance(self, other):
        ego_vehicle_transform = self._vehicle.get_transform()
        other_transform = other.get_transform()
        sign = -1 if self.is_ahead(other_transform) else 1

        ego_location = ego_vehicle_transform.location
        other_location = other_transform.location
        dx = ego_location.x - other_location.x
        dy = ego_location.y - other_location.y
        dz = ego_location.z - other_location.z
        target_vector = np.array([dx, dy, dz])
        norm_target = np.linalg.norm(target_vector)
        return sign * norm_target
    
    def get_front_vehicle_state(self):
        actor_list = self._world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        front_vehicles_info = []
        for target_vehicle in vehicle_list:
            # do not account for the ego vehicle
            if target_vehicle.id == self._vehicle.id:
                continue

            # if the object is not in our lane it's not an obstacle
            target_vehicle_waypoint = self._map.get_waypoint(target_vehicle.get_location())
            # consider absolute value of lane_id is enough for the experiment
            if abs(target_vehicle_waypoint.lane_id) != abs(ego_vehicle_waypoint.lane_id):
                continue
            # if target_vehicle_waypoint.road_id != ego_vehicle_waypoint.road_id or \
            #         target_vehicle_waypoint.lane_id != ego_vehicle_waypoint.lane_id:
            #     continue

            # get the state of front vehiles
            target_distance = self.get_distance(target_vehicle)
            target_speed = get_speed(target_vehicle)
            target_id = target_vehicle.id
            if target_distance > 0:
                front_vehicles_info.append((target_distance, target_speed, target_id))
        
        if front_vehicles_info:
            front_vehicles_info = sorted(front_vehicles_info, key=lambda x: x[0])
            return True, front_vehicles_info[0]
        else:
            return False, None

    def run_step(self, debug=False):
        # stop if no route to follow
        if not self._route:
            return emergency_stop()

        # purge obsolete waypoints in the route
        index = 0
        for i, waypoint in enumerate(self._route):
            if self.is_close(waypoint.transform):
                if debug:
                    print("Vehicle: {}".format(self._vehicle.get_transform()))
                    print("vehicle is close to obsolete waypoint: {}".format(waypoint))
                index += 1
            elif self.is_ahead(waypoint.transform):
                if debug:
                    print("Vehicle: {}".format(self._vehicle.get_transform()))
                    print("vehicle is ahead of obsolete waypoint: {}".format(waypoint))
                index += 1
            else:
                break
        if index != 0:
            self._route = self._route[index:]
            if debug:
                print("waypoint remain: {}".format(len(self._route)))
            if not self._route:
                return emergency_stop()

        # follow next waypoint
        target_waypoint = self._route[0]
        if debug:
            print("target waypoint: {}".format(target_waypoint))
            print("waypoint lane id: {}".format(target_waypoint.lane_id))

        has_front_vehicle, front_vehicle_state = self.get_front_vehicle_state()

        world_snapshot = self._world.get_snapshot()
        timestamp = world_snapshot.timestamp
        ego_vehicle_snapshot = world_snapshot.find(self._vehicle.id)
        front_vehicle_snapshot = None
        if has_front_vehicle:
            front_vehicle_distance, front_vehicle_speed, front_vehicle_id = front_vehicle_state
            print("obstacle {} distance: {:1f}m, speed: {:1f}km/h".format(
                front_vehicle_id, front_vehicle_distance, front_vehicle_speed))
            front_vehicle_snapshot = world_snapshot.find(front_vehicle_id)
            if front_vehicle_distance < self._driver.minimum_distance:
                return emergency_stop()
            self._driver.set_leader_info(front_vehicle_distance, front_vehicle_speed)
            desired_speed = self._driver.calc_desired_speed()
            print("target speed: {:1f}".format(desired_speed))
            control = self._vehicle_controller.run_step(desired_speed, target_waypoint)
        else:
            control = self._vehicle_controller.run_step(self._target_speed, target_waypoint)
        self.update_history(self.get_info_dict(timestamp, ego_vehicle_snapshot, front_vehicle_snapshot))
        
        return control
        

# ==============================================================================
# -- start simulation ----------------------------------------------------------
# ==============================================================================
@main
def simulation(debug=False):
    actor_list = []
    try:
        # Initialize
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)
        world = client.get_world()
        spectator = world.get_spectator()
        world_map = world.get_map()
        blueprint_library = world.get_blueprint_library()
        
        # Set up ego vehicle model and location
        vehicle_blueprint = random.choice(blueprint_library.filter('vehicle.tesla.*'))
        vehicle_blueprint.set_attribute('color', '255,255,255')
        spawn_point = carla.Transform(carla.Location(x=330, y=14, z=20), carla.Rotation(yaw=-180))

        # Create ego vehicle
        ego_vehicle = world.spawn_actor(vehicle_blueprint, spawn_point)
        actor_list.append(ego_vehicle)
        print("ego {} created!".format(ego_vehicle))

        # Set up leader vehicle model and location
        vehicle_blueprint = random.choice(blueprint_library.filter('vehicle.tesla.*'))
        vehicle_blueprint.set_attribute('color', '255,0,0')
        spawn_point = carla.Transform(carla.Location(x=310, y=14, z=20), carla.Rotation(yaw=-180))

        # Create leader vehicle
        leader_vehicle = world.spawn_actor(vehicle_blueprint, spawn_point)
        actor_list.append(leader_vehicle)
        print("leader {} created!".format(leader_vehicle))

        # Retrieve the closest waypoint.
        world.wait_for_tick()
        waypoint = world_map.get_waypoint(ego_vehicle.get_location())

        # Get route for vehicle to follow
        route = [waypoint]
        for _ in range(1000):
            waypoint = random.choice(waypoint.next(2.0))
            route.append(waypoint)

        print("Start: {}".format(route[0].transform))
        print("End: {}".format(route[-1].transform))
        draw_waypoints(world, route)

        # Disable physics, we're just teleporting the vehicle.
        ego_vehicle.set_simulate_physics(False)
        leader_vehicle.set_simulate_physics(False)

        # Teleport the vehicle at the starting point.
        ego_vehicle.set_transform(route[0].transform)
        leader_vehicle.set_transform(route[10].transform)

        # Turn on physics to apply control
        ego_vehicle.set_simulate_physics(True)
        leader_vehicle.set_simulate_physics(True)

        # set up agent to control ego vehicle
        agent = NaiveAgent(ego_vehicle, route, target_speed=60)
        leader_agent = NaiveAgent(leader_vehicle, route, target_speed=60)

        # Create leader vehicle speed profile
        speed_profile = []
        second_length = int(1 / TIME_INTERVAL)
        speed_profile.extend([10] * 5 * second_length)
        speed_profile.extend([20] * 5 * second_length)
        speed_profile.extend([30] * 5 * second_length)
        speed_profile.extend([40] * 5 * second_length)
        speed_profile.extend([50] * 5 * second_length)
        speed_profile.extend([60] * 5 * second_length)
        speed_profile.extend([50] * 5 * second_length)
        speed_profile.extend([40] * 5 * second_length)
        speed_profile.extend([20] * 5 * second_length)
        speed_profile.extend([10] * 5 * second_length)
        speed_profile.extend([0] * 30 * second_length)

        # workaround to make spectator follow ego vehicle
        world.wait_for_tick()
        dummy_bp = blueprint_library.find('sensor.other.collision')
        dummy_transform = carla.Transform(carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0))
        dummy = world.spawn_actor(dummy_bp, dummy_transform, 
            attach_to=ego_vehicle, attachment_type=carla.AttachmentType.SpringArm)
        actor_list.append(dummy)

        # get into synchronous mode to make sure time interval is guaranteed
        time.sleep(2.0)
        world.wait_for_tick()
        settings = world.get_settings()
        settings.synchronous_mode = True
        world.apply_settings(settings)
        settings.fixed_delta_seconds = TIME_INTERVAL

        while True:
            # synchronize with world
            world.tick()

            # follow ego vehicle
            spectator.set_transform(dummy.get_transform())

            # Apply control to leader vehicle
            if speed_profile:
                leader_agent.set_target_speed(speed_profile[0]) 
                speed_profile = speed_profile[1:]
            else:
                leader_agent.set_target_speed(25)
            control = leader_agent.run_step()
            leader_vehicle.apply_control(control)

            # Apply control to ego vehicle
            control = agent.run_step()
            if debug and control.throttle:
                print(control)
            ego_vehicle.apply_control(control)



    except KeyboardInterrupt:
        print("\nInterrupted")

    finally:
        print('destroying actors...')
        for actor in actor_list:
            if actor is not None:
                actor.destroy()

        # change back to asynchronous mode
        settings = world.get_settings()
        settings.synchronous_mode = False 
        world.apply_settings(settings)

        # save agent history
        agent.store_history()

        print('done.')
