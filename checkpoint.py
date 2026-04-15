import math
import pygame

import settings


_CHECKPOINT_CENTERS = [
    (523, 604),
    (603, 548),
    (580, 421),
    (361, 466),
    (310, 412),
    (370, 353),
    (573, 296),
    (600, 224),
    (560, 140),
    (225, 60),
    (131, 183),
    (318, 234),
    (287, 303),
    (100, 443),
    (153, 550),
    (281, 615),
]

CHECKPOINT_RADIUS = 14


def ordered_checkpoints():
    return list(_CHECKPOINT_CENTERS)


def car_center(x, y):
    return (
        x + settings.CAR_WIDTH / 2,
        y + settings.CAR_HEIGHT / 2,
    )


def reached_checkpoint(x, y, checkpoint_center):
    return distance_to_checkpoint(x, y, checkpoint_center) <= CHECKPOINT_RADIUS


def distance_to_checkpoint(x, y, checkpoint_center):
    cx, cy = car_center(x, y)
    tx, ty = checkpoint_center
    return math.hypot(cx - tx, cy - ty)


def draw_checkpoints(screen, font, checkpoints, *, show_indices=True):
    for i, (cx, cy) in enumerate(checkpoints):
        pygame.draw.circle(screen, (70, 170, 255), (cx, cy), CHECKPOINT_RADIUS, 1)
        if show_indices:
            label = font.render(str(i), True, (150, 220, 255))
            screen.blit(label, (cx + CHECKPOINT_RADIUS + 2, cy - CHECKPOINT_RADIUS))
