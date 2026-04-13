import pygame

class Track:
    def __init__(self):
        self.image = pygame.image.load("assets/track.png").convert()
        self.mask_image = pygame.image.load("assets/mask.png").convert()

        self.width = self.image.get_width()
        self.height = self.image.get_height()

    def draw(self, screen):
        screen.blit(self.image, (0,0))
    
    def is_on_road(self, x, y):
        x = int(x)
        y = int(y)
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        
        color = self.mask_image.get_at((x, y))
        return color.r > 200 and color.g > 200 and color.b > 200