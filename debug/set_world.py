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
# -- import helper function from local module --------------------------------------------
# ==============================================================================
from utils import main

# ==============================================================================
# -- start simulation ----------------------------------------------------------
# ==============================================================================
@main
def simulation():
    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)
    world = client.load_world('Town04')
    world.set_weather(carla.WeatherParameters.ClearNoon)