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
# -- import helper function from local module ----------------------------------
# ==============================================================================
from utils import TIME_INTERVAL, emergency_stop, get_speed
from IDM import IDM

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
        self._dt = TIME_INTERVAL
        # it works, but still has a little ossilation
        # args_lateral_dict = {
        #     'K_P': 0.08,
        #     'K_D': 0.01,
        #     'K_I': 0.04,
        #     'dt': self._dt}
        # args_longitudinal_dict = {
        #     'K_P': 1.0,
        #     'K_D': 0,
        #     'K_I': 0,
        #     'dt': self._dt}
        args_lateral_dict = {
            'K_P': 0.08,
            'K_D': 0.005,
            'K_I': 0.04,
            'dt': self._dt}
        args_longitudinal_dict = {
            'K_P': 1.0,
            'K_D': 0,
            'K_I': 0,
            'dt': self._dt}
        self._vehicle_controller = VehiclePIDController(self._vehicle,
            args_lateral=args_lateral_dict,
            args_longitudinal=args_longitudinal_dict)
    
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
           
            # get the state of front vehiles
            target_distance = self.get_distance(target_vehicle)
            target_speed = get_speed(target_vehicle)
            if target_distance > 0:
                front_vehicles_info.append((target_distance, target_speed))
        
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

        # find leader vehicle
        has_front_vehicle, front_vehicle_state = self.get_front_vehicle_state()
        if has_front_vehicle:
            front_vehicle_distance, front_vehicle_speed = front_vehicle_state
            print("obstacle distance: {:3f}m, speed: {:1f}km/h".format(
                front_vehicle_distance, front_vehicle_speed))
            if front_vehicle_distance < self._driver.minimum_distance:
                return emergency_stop()

            self._driver.set_leader_info(*front_vehicle_state)
            desired_speed = self._driver.calc_desired_speed()
            print("target speed: {:1f}".format(desired_speed))
            return self._vehicle_controller.run_step(desired_speed, target_waypoint)
        else:
            print("no obstacle, default speed: {}".format(self._target_speed))
            return self._vehicle_controller.run_step(self._target_speed, target_waypoint)
 