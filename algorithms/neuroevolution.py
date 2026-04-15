import colorsys
import math
import os
from pathlib import Path

import neat
import pygame

import checkpoint
import physics
import settings
from agent import Agent
from rendering import draw_car
from sensors import compute_sensor_readings
from ui import action_button_rect, draw_action_button, draw_status_hud


_ACTIONS = [
    (-0.90, 1),  # slight left + forward
    (0.0, 1),  # straight + forward
    (0.90, 1),  # slight right + forward
    (-1.80, 1),  # hard left + forward
    (1.80, 1),  # hard right + forward
]

_ACTION_LOGIT_PRIOR = (0.15, 0.14, 0.15, 0.14, 0.14)

_STALL_MOVEMENT_THRESHOLD = 0.5
_STALL_FRAME_LIMIT = 25
_MIN_PROGRESS_FRAMES = 45
_MIN_MEANINGFUL_PROGRESS = 26.0
_LAP_LINE_X = 340.0
_LAP_LINE_Y_MIN = 590.0
_LAP_LINE_Y_MAX = 650.0

_SPAWN_RADIUS_BASE = 10.0
_SPAWN_RADIUS_JITTER = 6.0

_NEAT_DEBUG = os.environ.get("NEAT_DEBUG", "").strip() in ("1", "true", "yes")
_NEAT_DEBUG_VERBOSE = os.environ.get("NEAT_DEBUG_VERBOSE", "").strip() in (
    "1",
    "true",
    "yes",
)


def build_net_inputs(state):
    return [
        state["speed"] / max(settings.NEAT_MAX_SPEED, 0.001),
        math.cos(state["angle"]),
        math.sin(state["angle"]),
        1.0 if state["front_near_on_road"] else 0.0,
        1.0 if state["front_left_on_road"] else 0.0,
        1.0 if state["front_right_on_road"] else 0.0,
        1.0 if state["left_on_road"] else 0.0,
        1.0 if state["right_on_road"] else 0.0,
        1.0 if state["front_far_on_road"] else 0.0,
    ]


def pick_action(raw_outputs):
    """Argmax over 5 forward actions with slight and hard steering choices."""
    n = len(_ACTIONS)
    outs = list(raw_outputs)[:n]
    while len(outs) < n:
        outs.append(0.0)
    biased = [outs[i] + _ACTION_LOGIT_PRIOR[i] for i in range(n)]
    idx = max(range(n), key=lambda j: biased[j])
    return idx, tuple(_ACTIONS[idx])


def spawn_position(i, n):
    radius = _SPAWN_RADIUS_BASE + (i % 5) * (_SPAWN_RADIUS_JITTER / 4.0)
    t = 2 * math.pi * i / max(n, 1) + (i * 0.31)
    return (
        settings.SPAWN_X + radius * math.cos(t),
        settings.SPAWN_Y + radius * math.sin(t),
    )


def agent_color(i, n):
    h = (i * 0.17 + 0.05) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def distance_from_start(x, y):
    return math.hypot(x - settings.SPAWN_X, y - settings.SPAWN_Y)


def compute_live_reward(*, speed, movement, crash, new_best_progress):
    if crash:
        return settings.NEAT_CRASH_PENALTY

    reward = settings.NEAT_REWARD_ALIVE
    reward += settings.NEAT_REWARD_FORWARD_SPEED_SCALE * max(0.0, speed)
    reward += settings.NEAT_REWARD_MOVEMENT_SCALE * movement
    if new_best_progress:
        reward += settings.NEAT_REWARD_PROGRESS_BONUS
    return reward


def checkpoint_progress_reward(prev_distance, current_distance):
    delta = prev_distance - current_distance
    if delta > 0.0:
        return settings.NEAT_REWARD_TO_CHECKPOINT_IMPROVEMENT * delta
    return -settings.NEAT_REWARD_TO_CHECKPOINT_BACKSLIDE * abs(delta)


def center_bias_reward(state):
    side_balance = 1.0 - abs(
        int(state["left_on_road"]) - int(state["right_on_road"])
    )
    diagonal_balance = 1.0 - abs(
        int(state["front_left_on_road"]) - int(state["front_right_on_road"])
    )
    return settings.NEAT_REWARD_CENTER_BONUS * (0.5 * side_balance + 0.5 * diagonal_balance)


def draw_sensor_debug(screen, x, y, angle, track):
    readings = compute_sensor_readings(
        x + settings.CAR_WIDTH / 2,
        y + settings.CAR_HEIGHT / 2,
        angle,
        track,
        settings.SENSOR_DISTANCE,
        settings.SENSOR_SIDE_ANGLE,
    )
    for name, sensor in readings.items():
        color = (0, 220, 90) if sensor["on_road"] else (220, 50, 50)
        radius = settings.SENSOR_DOT_RADIUS + (2 if name == "front_far" else 0)
        pygame.draw.circle(screen, color, (int(sensor["x"]), int(sensor["y"])), radius)


def make_agent(i, total_agents, genome, config, checkpoints):
    start_x, start_y = spawn_position(i, total_agents)
    start_distance = checkpoint.distance_to_checkpoint(start_x, start_y, checkpoints[0])
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    return Agent(
        x=start_x,
        y=start_y,
        angle=settings.INITIAL_ANGLE,
        speed=0.0,
        alive=True,
        fitness=0.0,
        color=agent_color(i, total_agents),
        net=net,
        frames_alive=0,
        last_x=start_x,
        last_y=start_y,
        stalled_frames=0,
        best_distance_from_start=distance_from_start(start_x, start_y),
        next_checkpoint_index=0,
        counted_lap_cycles=0,
        prev_distance_to_checkpoint=start_distance,
        death_reason=None,
    )


def crosses_lap_line(prev_x, prev_y, current_x, current_y):
    prev_cx, prev_cy = checkpoint.car_center(prev_x, prev_y)
    curr_cx, curr_cy = checkpoint.car_center(current_x, current_y)

    # Count only forward crossings through the start/finish segment.
    if prev_cx >= _LAP_LINE_X or curr_cx < _LAP_LINE_X:
        return False

    dx = curr_cx - prev_cx
    if dx <= 0.0:
        return False

    t = (_LAP_LINE_X - prev_cx) / dx
    if not 0.0 <= t <= 1.0:
        return False

    y_at_line = prev_cy + (curr_cy - prev_cy) * t
    return _LAP_LINE_Y_MIN <= y_at_line <= _LAP_LINE_Y_MAX


def build_eval_loop(screen, clock, track, font, button_font, session):
    def eval_genomes(genomes, config):
        session["generation"] += 1
        generation = session["generation"]
        total_agents = len(genomes)
        sim_frame = 0
        lap_count = 0
        checkpoints = checkpoint.ordered_checkpoints()
        agents = [
            make_agent(i, total_agents, genome, config, checkpoints)
            for i, (_, genome) in enumerate(genomes)
        ]

        running = True
        while running and any(a.alive for a in agents):
            sim_frame += 1
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    session["quit"] = True
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if action_button_rect(screen.get_width()).collidepoint(event.pos):
                        session["return_to_demo"] = True
                        running = False

            for idx, a in enumerate(agents):
                if not a.alive:
                    continue

                a.frames_alive += 1
                movement = 0.0

                st = physics.build_observation(
                    a.x, a.y, a.angle, a.speed, 0, track
                )
                raw = list(a.net.activate(build_net_inputs(st)))
                if len(raw) != len(_ACTIONS):
                    # Misconfigured genome vs config — avoid crashes, pad/trim
                    raw = (raw + [0.0] * len(_ACTIONS))[: len(_ACTIONS)]

                action_idx, action = pick_action(raw)
                a.last_x, a.last_y = a.x, a.y

                step_result = physics.step_car(
                    track,
                    a.x,
                    a.y,
                    a.angle,
                    a.speed,
                    0,
                    action,
                    respawn_on_crash=False,
                    steer_scale=settings.NEAT_STEER_SCALE,
                    friction=settings.NEAT_FRICTION,
                    max_speed=settings.NEAT_MAX_SPEED,
                )

                if step_result["crashed"]:
                    a.fitness += compute_live_reward(
                        speed=a.speed,
                        movement=0.0,
                        crash=True,
                        new_best_progress=False,
                    )
                    a.alive = False
                    a.death_reason = "crash"
                else:
                    a.x = step_result["x"]
                    a.y = step_result["y"]
                    a.angle = step_result["angle"]
                    a.speed = step_result["speed"]
                    movement = math.hypot(a.x - a.last_x, a.y - a.last_y)
                    current_distance = distance_from_start(a.x, a.y)
                    new_best_progress = current_distance > a.best_distance_from_start
                    if new_best_progress:
                        a.best_distance_from_start = current_distance

                    a.fitness += compute_live_reward(
                        speed=a.speed,
                        movement=movement,
                        crash=False,
                        new_best_progress=new_best_progress,
                    )
                    a.fitness += center_bias_reward(st)

                    # Reward progress only toward the correct next checkpoint.
                    next_idx = a.next_checkpoint_index % len(checkpoints)
                    current_cp = checkpoints[next_idx]
                    current_distance_to_checkpoint = checkpoint.distance_to_checkpoint(
                        a.x, a.y, current_cp
                    )
                    a.fitness += checkpoint_progress_reward(
                        a.prev_distance_to_checkpoint,
                        current_distance_to_checkpoint,
                    )

                    if checkpoint.reached_checkpoint(a.x, a.y, current_cp):
                        a.next_checkpoint_index += 1
                        a.fitness += settings.NEAT_REWARD_CHECKPOINT_BONUS
                        next_idx = a.next_checkpoint_index % len(checkpoints)
                        a.prev_distance_to_checkpoint = checkpoint.distance_to_checkpoint(
                            a.x, a.y, checkpoints[next_idx]
                        )
                    else:
                        a.prev_distance_to_checkpoint = current_distance_to_checkpoint

                    completed_cycles = a.next_checkpoint_index // len(checkpoints)
                    if (
                        completed_cycles > a.counted_lap_cycles
                        and crosses_lap_line(a.last_x, a.last_y, a.x, a.y)
                    ):
                        lap_count += 1
                        a.counted_lap_cycles = completed_cycles

                    if movement < _STALL_MOVEMENT_THRESHOLD:
                        a.stalled_frames += 1
                    else:
                        a.stalled_frames = 0

                    no_progress = (
                        a.frames_alive > _MIN_PROGRESS_FRAMES
                        and a.best_distance_from_start < _MIN_MEANINGFUL_PROGRESS
                    )
                    if a.stalled_frames >= _STALL_FRAME_LIMIT or no_progress:
                        a.fitness += settings.NEAT_STALL_DEATH_PENALTY
                        a.alive = False
                        a.death_reason = "stall"

                if _NEAT_DEBUG and idx == 0 and (
                    sim_frame % 30 == 0 or a.death_reason is not None
                ):
                    line = (
                        "[NEAT dbg] agent0 "
                        f"movement={movement:.3f} stalled_frames={a.stalled_frames} "
                        f"speed={a.speed:.3f} fitness={a.fitness:.2f} "
                        f"next_cp={a.next_checkpoint_index} "
                        f"dist_cp={a.prev_distance_to_checkpoint:.1f} "
                        f"death_reason={a.death_reason}"
                    )
                    if _NEAT_DEBUG_VERBOSE:
                        line += (
                            f" raw={[f'{x:.3f}' for x in raw]} "
                            f"idx={action_idx} action={action}"
                        )
                    print(line)

            track.draw(screen)
            if settings.SHOW_NEAT_CHECKPOINT_DEBUG:
                checkpoint.draw_checkpoints(screen, font, checkpoints)
            for a in agents:
                if not a.alive:
                    continue
                draw_car(
                    screen,
                    a.x,
                    a.y,
                    settings.CAR_WIDTH,
                    settings.CAR_HEIGHT,
                    a.angle,
                    a.color,
                    alpha=255,
                )
                if settings.SHOW_NEAT_SENSOR_DEBUG:
                    draw_sensor_debug(screen, a.x, a.y, a.angle, track)

            alive_n = sum(1 for a in agents if a.alive)
            lines = [
                f"Generation {generation}",
                f"Alive {alive_n}/{total_agents}",
                f"Lap {lap_count}",
            ]
            draw_status_hud(screen, font, lines)
            draw_action_button(
                screen,
                button_font,
                "Return",
                pygame.mouse.get_pos(),
            )

            pygame.display.flip()
            clock.tick(settings.FPS)

        for (_, genome), ag in zip(genomes, agents):
            genome.fitness = ag.fitness

    return eval_genomes


def default_config_path():
    return Path(__file__).resolve().parent.parent / "neat_config.ini"


def run_neat_live(screen, clock, track, max_generations=100, config_path=None):
    path = config_path or default_config_path()
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(path),
    )

    font = pygame.font.SysFont(None, 22)
    button_font = pygame.font.SysFont(None, 22)
    session = {
        "generation": 0,
        "return_to_demo": False,
        "quit": False,
    }

    def make_population():
        population = neat.Population(config)
        population.add_reporter(neat.StdOutReporter(True))
        population.add_reporter(neat.StatisticsReporter())
        return population

    population = make_population()
    for _ in range(max_generations):
        if session["quit"]:
            return "quit"
        if session["return_to_demo"]:
            return "demo"
        eval_loop = build_eval_loop(
            screen,
            clock,
            track,
            font,
            button_font,
            session,
        )
        population.run(eval_loop, 1)
        if session["quit"]:
            return "quit"
        if session["return_to_demo"]:
            return "demo"

    return "demo"
