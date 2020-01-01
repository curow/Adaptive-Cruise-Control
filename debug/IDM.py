import math

from utils import get_speed
from physical_constrains import get_vehicle_max_acceleration, get_vehicle_max_deceleration

# ==============================================================================
# -- Intelligent Driver Model --------------------------------------------------
# ==============================================================================
class IDM:
    def __init__(self, vehicle, desired_speed, dt):
        """
        desired_speed: [kmh]
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
        """
        leader_speed: [kmh]
        """
        # translate kmh to meters/second
        self.leader_distance = leader_distance
        self.leader_speed = leader_speed / 3.6
    
    def get_leader_info(self):
        """
        leader_speed: [kmh]
        """
        return self.leader_distance, self.leader_speed * 3.6

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
