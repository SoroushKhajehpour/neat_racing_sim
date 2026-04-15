import math

import settings
from collision import is_car_on_road
from sensors import compute_sensor_readings


def compute_reward(crashed, speed):
    if crashed:
        return settings.REWARD_CRASH
    forward = max(0.0, speed)
    return settings.REWARD_ALIVE + settings.REWARD_FORWARD_SPEED_SCALE * forward


def build_observation(x, y, angle, speed, crash_count, track):
    center_x = x + settings.CAR_WIDTH / 2
    center_y = y + settings.CAR_HEIGHT / 2
    readings = compute_sensor_readings(
        center_x,
        center_y,
        angle,
        track,
        settings.SENSOR_DISTANCE,
        settings.SENSOR_SIDE_ANGLE,
    )
    return {
        "x": x,
        "y": y,
        "angle": angle,
        "speed": speed,
        "crash_count": crash_count,
        "track": track,
        "front_near_on_road": readings["front_near"]["on_road"],
        "front_left_on_road": readings["front_left"]["on_road"],
        "front_right_on_road": readings["front_right"]["on_road"],
        "left_on_road": readings["left"]["on_road"],
        "right_on_road": readings["right"]["on_road"],
        "front_far_on_road": readings["front_far"]["on_road"],
    }


def _car_state(x, y, angle, speed, crash_count, reward, *, crashed, done):
    return {
        "x": x,
        "y": y,
        "angle": angle,
        "speed": speed,
        "crash_count": crash_count,
        "reward": reward,
        "crashed": crashed,
        "done": done,
    }


def step_car(
    track,
    x,
    y,
    angle,
    speed,
    crash_count,
    action,
    *,
    respawn_on_crash=True,
    steer_scale=1.0,
    friction=None,
    max_speed=None,
):
    steer, throttle = action

    next_angle = angle + steer * settings.TURN_SPEED * steer_scale
    next_speed = speed
    if throttle == 1:
        next_speed += settings.ACCELERATION
    elif throttle == -1:
        next_speed -= settings.BRAKE

    friction = settings.FRICTION if friction is None else friction
    max_speed = settings.MAX_SPEED if max_speed is None else max_speed

    next_speed *= friction
    next_speed = max(
        -settings.MAX_REVERSE_SPEED,
        min(max_speed, next_speed),
    )

    next_x = x + next_speed * math.cos(next_angle)
    next_y = y + next_speed * math.sin(next_angle)

    if is_car_on_road(
        next_x,
        next_y,
        settings.CAR_WIDTH,
        settings.CAR_HEIGHT,
        next_angle,
        track,
    ):
        reward = compute_reward(False, next_speed)
        return _car_state(
            next_x,
            next_y,
            next_angle,
            next_speed,
            crash_count,
            reward,
            crashed=False,
            done=False,
        )

    if respawn_on_crash:
        reward = compute_reward(True, 0.0)
        return _car_state(
            settings.SPAWN_X,
            settings.SPAWN_Y,
            settings.INITIAL_ANGLE,
            0.0,
            crash_count + 1,
            reward,
            crashed=True,
            done=True,
        )

    reward = compute_reward(True, speed)
    return _car_state(
        x,
        y,
        angle,
        speed,
        crash_count,
        reward,
        crashed=True,
        done=True,
    )
