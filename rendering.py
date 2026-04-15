import math
from pathlib import Path

import pygame

import settings

_ASSETS = Path(__file__).resolve().parent / "assets"
_CAR_SPRITE_PATH = _ASSETS / "racecar.png"
_BASE_CAR_IMAGE = None


def _get_base_car_image():
    global _BASE_CAR_IMAGE
    if _BASE_CAR_IMAGE is not None:
        return _BASE_CAR_IMAGE

    image = pygame.image.load(str(_CAR_SPRITE_PATH)).convert_alpha()
    src_w, src_h = image.get_size()

    target_w = 44
    scale = target_w / max(src_w, 1)
    target_h = max(1, int(src_h * scale))
    _BASE_CAR_IMAGE = pygame.transform.smoothscale(image, (target_w, target_h))
    return _BASE_CAR_IMAGE


def draw_car(
    screen,
    x,
    y,
    w,
    h,
    angle_rad,
    color,
    *,
    alpha=255,
):
    cx, cy = x + w / 2, y + h / 2

    car_surface = _get_base_car_image().copy()
    if alpha < 255:
        car_surface.set_alpha(alpha)

    # The sprite artwork points to the right, which matches angle 0 in the
    # physics code, so we rotate by the negative screen-space angle here.
    rotated = pygame.transform.rotate(car_surface, -math.degrees(angle_rad))
    rect = rotated.get_rect(center=(cx, cy))
    screen.blit(rotated, rect.topleft)

    if settings.SHOW_HITBOX:
        hitbox = pygame.Rect(int(x), int(y), int(w), int(h))
        pygame.draw.rect(screen, color, hitbox, width=1)
