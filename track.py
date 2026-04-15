from pathlib import Path

import pygame

_ASSETS = Path(__file__).resolve().parent / "assets"


def _load_image(path: Path):
    surf = pygame.image.load(str(path))
    try:
        return surf.convert()
    except pygame.error:
        return surf


class Track:
    def __init__(self):
        self.image = _load_image(_ASSETS / "track.png")
        self.mask_image = _load_image(_ASSETS / "mask.png")

        self.width = self.image.get_width()
        self.height = self.image.get_height()

    def draw(self, screen):
        screen.blit(self.image, (0, 0))

    def is_on_road(self, x, y):
        x = int(x)
        y = int(y)
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False

        color = self.mask_image.get_at((x, y))
        return color.r > 200 and color.g > 200 and color.b > 200