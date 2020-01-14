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
import inspect

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
# -- global constants  ---------------------------------------------------------
# ==============================================================================
TIME_INTERVAL = 0.05

# ==============================================================================
# -- utility function ----------------------------------------------------------
# ==============================================================================
def main(fn):
    """Call fn with command line arguments. Used as a decorator.
    The main decorator marks the function that starts a program. For example,

    @main
    def my_run_function():
        # function body

    Use this instead of the typical __name__ == "__main__" predicate.
    """
    if inspect.stack()[1][0].f_locals['__name__'] == '__main__':
        args = sys.argv[1:] # Discard the script name from command line
        fn(*args)
    return fn

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
    elif unit == 'ms':
        return vel_ms
    else:
        raise Exception("Illegal Argument")

def emergency_stop():
    return carla.VehicleControl(
        steer=0,
        throttle=0,
        brake=1.0,
        hand_brake=False,
        manual_gear_shift=False
    )
