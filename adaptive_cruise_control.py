# ==============================================================================
# -- necessary module import -------------------------------
# ==============================================================================
import sys
import os
import glob
import random
import time
import math
import numpy as np

# ==============================================================================
# -- find carla module and add PythonAPI to path -------------------------------
# ==============================================================================
try:
    python_api_path = '/home/hollow/carla/CARLA_0.9.6/PythonAPI/carla/'
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
# -- import staff from local file ----------------------------------------------
# ==============================================================================
from utils import main

# ==============================================================================
# -- agent class ---------------------------------------------------------------
# ==============================================================================
class NaiveAgent:
    def __init__(self, vehicle, route, target_speed=20):
        self._vehicle = vehicle
        self._route = route
        self._target_speed = target_speed
        self._world = self._vehicle.get_world()
        self._map = self._world.get_map()
        self._vehicle_controller = VehiclePIDController(self._vehicle)

    def emergency_stop(self):
        return carla.VehicleControl(
            steer=0,
            throttle=0,
            brake=1.0,
            hand_brake=False,
            manual_gear_shift=False
        )

    def is_close(self, waypoint, epsilon=0.1):
        vehicle_transform = self._vehicle.get_transform()
        loc = vehicle_transform.location
        dx = waypoint.transform.location.x - loc.x
        dy = waypoint.transform.location.y - loc.y

        return np.linalg.norm(np.array([dx, dy])) < epsilon
    
    def is_ahead(self, waypoint):
        vehicle_transform = self._vehicle.get_transform()
        loc = vehicle_transform.location
        orientation = vehicle_transform.rotation.yaw

        dx = waypoint.transform.location.x - loc.x
        dy = waypoint.transform.location.y - loc.y
        target_vector = np.array([dx, dy])
        norm_target = np.linalg.norm(target_vector)

        forward_vector = np.array([
            math.cos(math.radians(orientation)),
            math.sin(math.radians(orientation))
            ])
        d_angle = math.degrees(math.acos(np.dot(forward_vector, target_vector) / norm_target))

        return d_angle > 90

    def run_step(self, debug=False):
        # stop if no route to follow
        if not self._route:
            return self.emergency_stop()

        # purge obsolete waypoints in the route
        index = 0
        for i, waypoint in enumerate(self._route):
            if self.is_close(waypoint):
                if debug:
                    print("Vehicle: {}".format(self._vehicle.get_transform()))
                    print("vehicle is close to obsolete waypoint: {}".format(waypoint))
                index += 1
            elif self.is_ahead(waypoint):
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
                return self.emergency_stop()

        # follow next waypoint
        target_waypoint = self._route[0]
        if debug:
            print("target waypoint: {}".format(target_waypoint))

        return self._vehicle_controller.run_step(self._target_speed, target_waypoint)
        
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
        world = client.load_world('Town04')
        world.set_weather(carla.WeatherParameters.ClearNoon)
        spectator = world.get_spectator()
        world_map = world.get_map()
        blueprint_library = world.get_blueprint_library()
        world.set_weather(carla.WeatherParameters.ClearNoon)
        
        # Set up ego vehicle model and location
        vehicle_blueprint = random.choice(blueprint_library.filter('vehicle.tesla.*'))
        # spawn_point = random.choice(world_map.get_spawn_points())
        spawn_point = carla.Transform(carla.Location(x=330, y=14, z=20), carla.Rotation(yaw=-180))

        # Create ego vehicle
        ego_vehicle = world.spawn_actor(vehicle_blueprint, spawn_point)
        actor_list.append(ego_vehicle)
        print("{} created!".format(ego_vehicle))

        # Wait for world to get the vehicle actor
        world.tick()
        world_snapshot = world.wait_for_tick()
        actor_snapshot = world_snapshot.find(ego_vehicle.id)
        
        # Set spectator at given transform (vehicle transform)
        spectator.set_transform(actor_snapshot.get_transform())
        
        # Retrieve the closest waypoint.
        waypoint = world_map.get_waypoint(ego_vehicle.get_location())

        # Get route for vehicle to follow
        route = [waypoint]
        for _ in range(250):
            waypoint = random.choice(waypoint.next(2.0))
            route.append(waypoint)

        print("Start: {}".format(route[0].transform))
        print("End: {}".format(route[-1].transform))
        draw_waypoints(world, route)

        # Disable physics, we're just teleporting the vehicle.
        ego_vehicle.set_simulate_physics(False)

        # Teleport the vehicle at the starting point.
        ego_vehicle.set_transform(route[0].transform)

        # Wait for vehicle to land on ground
        time.sleep(2.0)

        # Turn on physics to apply control
        ego_vehicle.set_simulate_physics(True)

        # set up agent to control ego vehicle
        agent = NaiveAgent(ego_vehicle, route, target_speed=30)
        while True:
            # Wait for world to get ready
            world.wait_for_tick(10.0)
            # Apply control
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
        print('done.')