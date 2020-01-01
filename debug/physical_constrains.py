# ==============================================================================
# -- Vehicle Physical Constrains -----------------------------------------------
# ==============================================================================
# https://github.com/carla-simulator/ros-bridge/blob/master/
# carla_ackermann_control/src/carla_ackermann_control/carla_control_physics.py

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
