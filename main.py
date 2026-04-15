import argparse
import os
import sys

import pygame

import settings
from algorithms import human
from algorithms.neuroevolution import run_neat_live
from env import RacingEnv
from track import Track
from ui import action_button_rect, draw_action_button

WINDOW_SIZE = (720, 720)


def run_human_mode(neat_generations: int = 100) -> None:
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    clock = pygame.time.Clock()
    button_font = pygame.font.SysFont(None, 22)

    env = RacingEnv()
    state = env.reset()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if action_button_rect(screen.get_width()).collidepoint(event.pos):
                    next_mode = run_neat_live(
                        screen,
                        clock,
                        env.track,
                        max_generations=neat_generations,
                    )
                    if next_mode == "demo":
                        state = env.reset()
                    else:
                        running = False
        if not running:
            break

        action = human.get_action(state)
        state, _, done = env.step(action)

        env.render(screen)
        if done:
            state = env.reset()

        draw_action_button(
            screen,
            button_font,
            "Run Neuroevolution",
            pygame.mouse.get_pos(),
        )

        pygame.display.flip()
        clock.tick(settings.FPS)

    pygame.quit()


def run_neat(max_generations: int) -> None:
    next_mode = "quit"
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    clock = pygame.time.Clock()
    track = Track()
    try:
        next_mode = run_neat_live(screen, clock, track, max_generations=max_generations)
    finally:
        pygame.quit()

    if next_mode == "demo":
        run_human_mode(neat_generations=max_generations)


def main(argv=None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    default_mode = os.environ.get("RACING_MODE", "demo").lower()
    if default_mode not in ("demo", "neat"):
        default_mode = "demo"

    parser = argparse.ArgumentParser(description="AI racing sim: demo or NEAT mode.")
    parser.add_argument(
        "mode",
        nargs="?",
        default=default_mode,
        choices=("demo", "neat"),
        help="demo: single car + driver; neat: live neuroevolution (default: %(default)s or RACING_MODE)",
    )
    parser.add_argument(
        "-g",
        "--generations",
        type=int,
        default=100,
        dest="neat_generations",
        metavar="N",
        help="neat mode only: max generations (default: 100)",
    )
    args = parser.parse_args(argv)

    if args.mode == "demo":
        run_human_mode(neat_generations=args.neat_generations)
    else:
        run_neat(max_generations=args.neat_generations)


if __name__ == "__main__":
    main()
