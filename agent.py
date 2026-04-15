from dataclasses import dataclass
from typing import Any, Optional, Tuple


@dataclass
class Agent:
    x: float
    y: float
    angle: float
    speed: float
    alive: bool = True
    fitness: float = 0.0
    color: Tuple[int, int, int] = (220, 50, 50)
    net: Optional[Any] = None

    frames_alive: int = 0
    last_x: float = 0.0
    last_y: float = 0.0
    stalled_frames: int = 0
    best_distance_from_start: float = 0.0
    next_checkpoint_index: int = 0
    counted_lap_cycles: int = 0
    prev_distance_to_checkpoint: float = 0.0
    death_reason: Optional[str] = None
