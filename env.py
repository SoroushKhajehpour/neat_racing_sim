import math

import pygame

import settings
import physics
from rendering import draw_car
from sensors import compute_sensor_readings
from track import Track


class RacingEnv:
    def __init__(self):
        self.track = Track()
        self.x = settings.SPAWN_X
        self.y = settings.SPAWN_Y
        self.angle = settings.INITIAL_ANGLE
        self.speed = 0.0
        self.crash_count = 0
        self._state = {}
        self._last_reward = 0.0
        self.font = pygame.font.SysFont(None, 24)

    def _build_state(self):
        return physics.build_observation(
            self.x,
            self.y,
            self.angle,
            self.speed,
            self.crash_count,
            self.track,
        )

    def _load_step_result(self, step_result):
        self.x = step_result["x"]
        self.y = step_result["y"]
        self.angle = step_result["angle"]
        self.speed = step_result["speed"]
        self.crash_count = step_result["crash_count"]
        self._last_reward = step_result["reward"]
        self._state = self._build_state()

    def _sensor_readings(self):
        center_x = self.x + settings.CAR_WIDTH / 2
        center_y = self.y + settings.CAR_HEIGHT / 2
        return compute_sensor_readings(
            center_x,
            center_y,
            self.angle,
            self.track,
            settings.SENSOR_DISTANCE,
            settings.SENSOR_SIDE_ANGLE,
        )

    def reset(self):
        self.x = settings.SPAWN_X
        self.y = settings.SPAWN_Y
        self.angle = settings.INITIAL_ANGLE
        self.speed = 0.0
        self.crash_count = 0
        self._last_reward = 0.0
        self._state = self._build_state()
        return self._state

    def step(self, action):
        step_result = physics.step_car(
            self.track,
            self.x,
            self.y,
            self.angle,
            self.speed,
            self.crash_count,
            action,
            respawn_on_crash=True,
        )
        self._load_step_result(step_result)
        return self._state, step_result["reward"], step_result["done"]

    def render(self, screen, driver_name="driver", *, show_hud=False, show_sensors=False):
        self.track.draw(screen)
        draw_car(
            screen,
            self.x,
            self.y,
            settings.CAR_WIDTH,
            settings.CAR_HEIGHT,
            self.angle,
            (220, 50, 50),
        )

        if show_sensors:
            self._draw_sensors(screen, self._sensor_readings(), settings.SENSOR_DOT_RADIUS)

        if show_hud:
            self._draw_hud(screen, driver_name)

    def _draw_hud(self, screen, driver_name):
        hud_lines = [
            f"driver: {driver_name}",
            f"x: {self.x:.1f}  y: {self.y:.1f}",
            f"speed: {self.speed:.2f}",
            f"angle: {math.degrees(self.angle):.1f} deg",
            f"crashes (episode): {self.crash_count}",
            f"reward: {self._last_reward:.3f}",
            f"obs sensors  FN:{self._state.get('front_near_on_road', False)}  "
            f"FL:{self._state.get('front_left_on_road', False)}  "
            f"FR:{self._state.get('front_right_on_road', False)}",
            f"L:{self._state.get('left_on_road', False)}  "
            f"R:{self._state.get('right_on_road', False)}  "
            f"FF:{self._state.get('front_far_on_road', False)}",
        ]
        for i, line in enumerate(hud_lines):
            text = self.font.render(line, True, (255, 255, 255))
            screen.blit(text, (10, 10 + i * 26))

    @staticmethod
    def _draw_sensors(screen, readings, radius):
        for key in readings:
            s = readings[key]
            color = (0, 200, 60) if s["on_road"] else (220, 40, 40)
            dot_radius = radius + 2 if key == "front_far" else radius
            pygame.draw.circle(screen, color, (int(s["x"]), int(s["y"])), dot_radius)
