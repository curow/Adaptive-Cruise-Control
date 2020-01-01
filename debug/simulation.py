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

# ==============================================================================
# -- import helper function from local module ----------------------------------
# ==============================================================================
from utils import main, draw_waypoints, TIME_INTERVAL
from naive_agent import NaiveAgent

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
        waypoint = world_map.get_waypoint(carla.Location(x=330, y=14, z=20))

        # Get route for vehicle to follow
        route = [waypoint]
        for _ in range(500):
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

        # set up agent to control ego and leader vehicle
        # set ego vehicle highest speed to 60 in order to follow leader vehicle
        agent = NaiveAgent(ego_vehicle, route, target_speed=60)
        leader_agent = NaiveAgent(leader_vehicle, route[10:])

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
        speed_profile.extend([0] * 10 * second_length)

        # workaround to make spectator follow ego vehicle
        world.wait_for_tick()
        dummy_bp = blueprint_library.find('sensor.other.collision')
        dummy_transform = carla.Transform(carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0))
        dummy = world.spawn_actor(dummy_bp, dummy_transform, 
            attach_to=ego_vehicle, attachment_type=carla.AttachmentType.SpringArm)
        actor_list.append(dummy)

        # get into synchronous mode to make sure time interval is guaranteed
        world.wait_for_tick()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = TIME_INTERVAL
        world.apply_settings(settings)

        while True:
            # synchronize with world
            world.tick()

            # get time
            world_snapshot = world.get_snapshot()
            timestamp = world_snapshot.timestamp
            # print("frame number: {}, spend {} seconds since last tick, total time spend: {}".format(
            #     timestamp.frame, timestamp.delta_seconds, timestamp.elapsed_seconds))

            # follow ego vehicle
            spectator.set_transform(dummy.get_transform())

            # compute control to leader vehicle
            if speed_profile:
                print("speed profile[0]: {}".format(speed_profile[0]))
                leader_agent.set_target_speed(speed_profile[0]) 
                speed_profile = speed_profile[1:]
            else:
                leader_agent.set_target_speed(25)
            
            print("leader vehicle:")
            leader_control = leader_agent.run_step()

            # compute control to ego vehicle
            print("ego vehicle:")
            ego_control = agent.run_step()
            if debug and control.throttle:
                print(control)

            # aplly control
            ego_vehicle.apply_control(ego_control)
            leader_vehicle.apply_control(leader_control)



    except KeyboardInterrupt:
        print("\nInterrupted")

    finally:
        # cleanup
        print('destroying actors...')
        for actor in actor_list:
            if actor is not None:
                actor.destroy()

        # change back to asynchronous mode
        settings = world.get_settings()
        settings.synchronous_mode = False 
        world.apply_settings(settings)
        print('done.')
