"""Grid-backed environment fields."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from animalcula.sim.types import Vec2


@dataclass(slots=True)
class Grid2D:
    width: float
    height: float
    resolution: float
    cols: int = field(init=False)
    rows: int = field(init=False)
    values: list[float] = field(init=False)

    def __post_init__(self) -> None:
        self.cols = int(self.width / self.resolution)
        self.rows = int(self.height / self.resolution)
        self.values = [0.0] * (self.cols * self.rows)

    def index_for_position(self, position: Vec2) -> tuple[int, int]:
        wrapped_x = position.x % self.width
        wrapped_y = position.y % self.height
        col = min(int(wrapped_x / self.resolution), self.cols - 1)
        row = min(int(wrapped_y / self.resolution), self.rows - 1)
        return col, row

    def sample(self, position: Vec2) -> float:
        col, row = self.index_for_position(position)
        return self.values[(row * self.cols) + col]

    def set_value(self, col: int, row: int, value: float) -> None:
        self.values[(row * self.cols) + col] = value

    def position_for_cell(self, col: int, row: int) -> Vec2:
        return Vec2(
            x=(col + 0.5) * self.resolution,
            y=(row + 0.5) * self.resolution,
        )

    def fill_light_gradient(self, direction: tuple[float, float], intensity: float) -> None:
        direction_length = math.hypot(*direction)
        if direction_length == 0.0:
            normalized = (1.0, 0.0)
        else:
            normalized = (direction[0] / direction_length, direction[1] / direction_length)

        for row in range(self.rows):
            y = ((row + 0.5) * self.resolution) / self.height
            for col in range(self.cols):
                x = ((col + 0.5) * self.resolution) / self.width
                projection = (x * normalized[0]) + (y * normalized[1])
                value = max(0.0, min(projection, 1.0)) * intensity
                self.set_value(col=col, row=row, value=value)
