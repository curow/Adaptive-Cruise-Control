# ==============================================================================
# -- necessary module import -------------------------------
# ==============================================================================
import sys
import os
import glob
import random
import time
import math

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

# ==============================================================================
# -- utility function ----------------------------------------------------------
# ==============================================================================
def draw_waypoints(world, waypoints, z=3):
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

    # Get waypoints for vehicle to follow
    waypoints = [waypoint]
    for _ in range(500):
        # Find next waypoint 1 meters ahead.
        waypoint = random.choice(waypoint.next(2.0))
        waypoints.append(waypoint)
    print("Start: {}".format(waypoints[0].transform))
    print("End: {}".format(waypoints[-1].transform))
    draw_waypoints(world, waypoints)

    # Disable physics, we're just teleporting the vehicle.
    ego_vehicle.set_simulate_physics(False)

    # Teleport the vehicle at the starting point.
    ego_vehicle.set_transform(waypoints[0].transform)

    # Wait for vehicle to land on ground
    time.sleep(4.0)

    # Turn on physics to apply control
    ego_vehicle.set_simulate_physics(True)
    while True:
        # Wait for world to ready
        world.wait_for_tick(10.0)
        # Apply random control
        throttle = random.uniform(0, 1)
        control = carla.VehicleControl(
            throttle=throttle,
            manual_gear_shift=False)
        ego_vehicle.apply_control(control)


except KeyboardInterrupt:
    print("\nInterrupted")

finally:
    print('destroying actors...')
    for actor in actor_list:
        if actor is not None:
            actor.destroy()
    print('done.')