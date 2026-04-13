def move_is_safe(x, y, width, height, dx, dy, track):
    new_x = x + dx
    new_y = y + dy

    points = [
        (new_x, new_y),
        (new_x + width, new_y),
        (new_x, new_y + height),
        (new_x + width, new_y + height),
        (new_x + width // 2, new_y + height // 2),
    ]

    for px, py in points:
        if not track.is_on_road(px, py):
            return False

    return True


def get_action(x, y, width, height, track, speed):
    candidates = [
        (speed, 0),    # right
        (0, -speed),   # up
        (0, speed),    # down
        (-speed, 0),   # left
    ]

    for dx, dy in candidates:
        if move_is_safe(x, y, width, height, dx, dy, track):
            return (dx, dy)

    return (0, 0)