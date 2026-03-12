"""Core runtime types for the simulation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


class NodeType(str, Enum):
    BODY = "body"
    MOUTH = "mouth"
    PHOTORECEPTOR = "photoreceptor"


@dataclass(slots=True, frozen=True)
class Vec2:
    x: float
    y: float

    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Vec2":
        return Vec2(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar: float) -> "Vec2":
        return Vec2(self.x / scalar, self.y / scalar)

    def magnitude(self) -> float:
        return math.hypot(self.x, self.y)

    def normalized(self) -> "Vec2":
        length = self.magnitude()
        if length == 0.0:
            return Vec2.zero()
        return self / length

    @classmethod
    def zero(cls) -> "Vec2":
        return cls(0.0, 0.0)


@dataclass(slots=True, frozen=True)
class NodeState:
    position: Vec2
    velocity: Vec2
    accumulated_force: Vec2
    drag_coeff: float
    radius: float
    node_type: NodeType = NodeType.BODY


@dataclass(slots=True, frozen=True)
class EdgeState:
    a: int
    b: int
    rest_length: float
    stiffness: float
    has_motor: bool = False
    motor_strength: float = 0.0


@dataclass(slots=True, frozen=True)
class BrainState:
    input_weights: tuple[tuple[float, ...], ...]
    recurrent_weights: tuple[tuple[float, ...], ...]
    biases: tuple[float, ...]
    time_constants: tuple[float, ...]
    states: tuple[float, ...]
    output_size: int


@dataclass(slots=True, frozen=True)
class CreatureState:
    node_indices: tuple[int, ...]
    energy: float
    brain: BrainState | None = None
    last_sensed_inputs: tuple[float, ...] = ()
    last_brain_outputs: tuple[float, ...] = ()
