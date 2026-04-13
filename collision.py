import math


def car_center(x, y, w, h):
    return x + w / 2, y + h / 2


def body_sample_points(cx, cy, w, h, angle_rad):

    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    half_w = w / 2
    half_h = h / 2
    local = [
        (-half_w, -half_h),
        (half_w, -half_h),
        (-half_w, half_h),
        (half_w, half_h),
    ]
    points = [(cx, cy)]
    for lx, ly in local:
        px = cx + lx * cos_a - ly * sin_a
        py = cy + lx * sin_a + ly * cos_a
        points.append((px, py))
    return points


def is_car_on_road(x, y, w, h, angle_rad, track):
 
    cx, cy = car_center(x, y, w, h)
    for px, py in body_sample_points(cx, cy, w, h, angle_rad):
        if not track.is_on_road(px, py):
            return False
    return True
