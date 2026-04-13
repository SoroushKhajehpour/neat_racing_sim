import math

import pygame

import settings
from collision import is_car_on_road
from track import Track


def draw_car(screen, x, y, w, h, angle_rad):
 
    cx, cy = x + w / 2, y + h / 2
    car_surface = pygame.Surface((w, h), pygame.SRCALPHA)
    car_surface.fill((220, 50, 50))
    # Surface is drawn along +x; rotate so that axis aligns with (cos(angle), sin(angle)).
    rotated = pygame.transform.rotate(car_surface, -math.degrees(angle_rad))
    rect = rotated.get_rect(center=(cx, cy))
    screen.blit(rotated, rect)


pygame.init()
screen = pygame.display.set_mode((720, 720))
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 26)

track = Track()

x = settings.SPAWN_X
y = settings.SPAWN_Y
angle = settings.INITIAL_ANGLE
speed = 0.0
crash_count = 0

running = True
while running: 

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        speed += settings.ACCELERATION
    if keys[pygame.K_DOWN]:
        speed -= settings.BRAKE
    if keys[pygame.K_LEFT]:
        angle -= settings.TURN_SPEED
    if keys[pygame.K_RIGHT]:
        angle += settings.TURN_SPEED

    speed *= settings.FRICTION
    speed = max(-settings.MAX_REVERSE_SPEED, min(settings.MAX_SPEED, speed))

    vx = speed * math.cos(angle)
    vy = speed * math.sin(angle)
    x_new = x + vx
    y_new = y + vy

    if is_car_on_road(
        x_new,
        y_new,
        settings.CAR_WIDTH,
        settings.CAR_HEIGHT,
        angle,
        track,
    ):
        x, y = x_new, y_new
    else:
        x = settings.SPAWN_X
        y = settings.SPAWN_Y
        angle = settings.INITIAL_ANGLE
        speed = 0.0
        crash_count += 1

    track.draw(screen)
    draw_car(screen, x, y, settings.CAR_WIDTH, settings.CAR_HEIGHT, angle)

    hud_lines = [
        f"x: {x:.1f}  y: {y:.1f}",
        f"speed: {speed:.2f}",
        f"angle: {math.degrees(angle):.1f} deg",
        f"crashes: {crash_count}",
    ]
    for i, line in enumerate(hud_lines):
        text = font.render(line, True, (255, 255, 255))
        screen.blit(text, (10, 10 + i * 28))

    pygame.display.flip()
    clock.tick(settings.FPS)

pygame.quit()
