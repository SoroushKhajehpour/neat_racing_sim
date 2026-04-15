import math


def compute_sensor_readings(cx, cy, angle, track, distance, side_angle):
    sensor_angles = {
        "front_near": angle,
        "front_left": angle - side_angle,
        "front_right": angle + side_angle,
        "left": angle - math.pi / 2,
        "right": angle + math.pi / 2,
    }
    readings = {}
    for name, sensor_angle in sensor_angles.items():
        sensor_x = cx + distance * math.cos(sensor_angle)
        sensor_y = cy + distance * math.sin(sensor_angle)
        readings[name] = {
            "x": sensor_x,
            "y": sensor_y,
            "on_road": track.is_on_road(sensor_x, sensor_y),
        }

    far_front_distance = distance * 2.2
    far_front_x = cx + far_front_distance * math.cos(angle)
    far_front_y = cy + far_front_distance * math.sin(angle)
    readings["front_far"] = {
        "x": far_front_x,
        "y": far_front_y,
        "on_road": track.is_on_road(far_front_x, far_front_y),
    }
    return readings
