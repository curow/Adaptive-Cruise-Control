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
from agents.navigation.local_planner import LocalPlanner
# ==============================================================================
# -- import helper function from local module --------------------------------------------
# ==============================================================================
from utils import main

# ==============================================================================
# --global constants  ---------------------------------------------------------------------------------------------------
# ==============================================================================
from adaptive_cruise_control import TIME_INTERVAL

# ==============================================================================
# -- Leader Agent--------------------------------------------
# ==============================================================================
class LeaderAgent:
    def __init__(self, vehicle, target_speed=25):
        self._vehicle = vehicle
        self._local_planner = LocalPlanner(self._vehicle)
        self._target_speed = target_speed
        self._speed_profile = []
    
    def set_speed_profile(self, speed_profile):
        self._speed_profile = speed_profile

    def run_step(self):
        if self._speed_profile:
            target_speed = self._speed_profile[0]
            print("speed :{}".format(target_speed))
            self.set_speed_profile(self._speed_profile[1:])
        else:
            target_speed = self._target_speed
        self._local_planner.set_speed(target_speed)
        return self._local_planner.run_step()

# ==============================================================================
# -- start simulation ----------------------------------------------------------
# ==============================================================================
@main
def simulation(debug=True):
    actor_list = []
    try:
        # Initialize
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)
        world = client.get_world()
        world_map = world.get_map()
        blueprint_library = world.get_blueprint_library()
        
        # Set up ego vehicle model and location
        vehicle_blueprint = random.choice(blueprint_library.filter('vehicle.tesla.*'))
        vehicle_blueprint.set_attribute('color', '255,0,0')
        spawn_point = carla.Transform(carla.Location(x=310, y=14, z=20), carla.Rotation(yaw=-180))

        # Create ego vehicle
        ego_vehicle = world.spawn_actor(vehicle_blueprint, spawn_point)
        actor_list.append(ego_vehicle)
        print("{} created!".format(ego_vehicle))

        # Create ego vehicle speed profile
        speed_profile = []
        second_length = int(1 / TIME_INTERVAL)
        speed_profile.extend([5] * 5 * second_length)
        speed_profile.extend([10] * 5 * second_length)
        speed_profile.extend([15] * 5 * second_length)
        speed_profile.extend([20] * 5 * second_length)

        # set up agent to control ego vehicle
        agent = LeaderAgent(ego_vehicle, target_speed=25)
        agent.set_speed_profile(speed_profile)

        while True:
            # synchronize with world
            world.wait_for_tick()

            # Apply control
            control = agent.run_step()
            ego_vehicle.apply_control(control)
        
    except KeyboardInterrupt:
            print("\nLeader Interrupted")

    finally:
        print('destroying leader actors...')
        for actor in actor_list:
            if actor is not None:
                actor.destroy()

        print('leader done.')


        