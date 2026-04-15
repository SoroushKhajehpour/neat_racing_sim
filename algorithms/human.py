import pygame


def get_action(_state):
    keys = pygame.key.get_pressed()
    steer = 0
    if keys[pygame.K_LEFT] and not keys[pygame.K_RIGHT]:
        steer = -1
    elif keys[pygame.K_RIGHT] and not keys[pygame.K_LEFT]:
        steer = 1

    throttle = 0
    if keys[pygame.K_UP] and not keys[pygame.K_DOWN]:
        throttle = 1
    elif keys[pygame.K_DOWN] and not keys[pygame.K_UP]:
        throttle = -1

    return steer, throttle
