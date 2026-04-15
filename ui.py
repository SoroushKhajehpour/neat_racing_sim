import pygame

ACTION_BUTTON_W = 216
ACTION_BUTTON_H = 38
ACTION_BUTTON_MARGIN = 16
HUD_PANEL_PADDING_X = 12
HUD_PANEL_PADDING_Y = 10
HUD_LINE_GAP = 6


def action_button_rect(screen_w: int) -> pygame.Rect:
    return pygame.Rect(
        screen_w - ACTION_BUTTON_W - ACTION_BUTTON_MARGIN,
        ACTION_BUTTON_MARGIN,
        ACTION_BUTTON_W,
        ACTION_BUTTON_H,
    )


def draw_action_button(screen, font, text: str, mouse_pos) -> pygame.Rect:
    rect = action_button_rect(screen.get_width())
    hover = rect.collidepoint(mouse_pos)
    bg = (40, 40, 40) if hover else (12, 12, 12)
    border = (235, 235, 235)

    pygame.draw.rect(screen, bg, rect, border_radius=10)
    pygame.draw.rect(screen, border, rect, 2, border_radius=10)

    surf = font.render(text, True, (255, 255, 255))
    screen.blit(surf, surf.get_rect(center=rect.center))
    return rect


def draw_status_hud(screen, font, lines) -> pygame.Rect:
    rendered = [font.render(line, True, (255, 255, 255)) for line in lines]
    width = max((surf.get_width() for surf in rendered), default=0)
    height = sum(surf.get_height() for surf in rendered)
    if rendered:
        height += HUD_LINE_GAP * (len(rendered) - 1)

    rect = pygame.Rect(
        16,
        16,
        width + HUD_PANEL_PADDING_X * 2,
        height + HUD_PANEL_PADDING_Y * 2,
    )
    panel = pygame.Surface(rect.size, pygame.SRCALPHA)
    panel.fill((0, 0, 0, 170))
    screen.blit(panel, rect.topleft)

    y = rect.y + HUD_PANEL_PADDING_Y
    for surf in rendered:
        screen.blit(surf, (rect.x + HUD_PANEL_PADDING_X, y))
        y += surf.get_height() + HUD_LINE_GAP

    return rect
