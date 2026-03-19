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
    boundary: str = "toroidal"
    cols: int = field(init=False)
    rows: int = field(init=False)
    values: list[float] = field(init=False)

    def __post_init__(self) -> None:
        self.cols = int(self.width / self.resolution)
        self.rows = int(self.height / self.resolution)
        self.values = [0.0] * (self.cols * self.rows)

    def index_for_position(self, position: Vec2) -> tuple[int, int]:
        if self.boundary == "bounded":
            clamped_x = max(0.0, min(position.x, self.width - 1e-9))
            clamped_y = max(0.0, min(position.y, self.height - 1e-9))
            col = min(int(clamped_x / self.resolution), self.cols - 1)
            row = min(int(clamped_y / self.resolution), self.rows - 1)
        else:
            wrapped_x = position.x % self.width
            wrapped_y = position.y % self.height
            col = min(int(wrapped_x / self.resolution), self.cols - 1)
            row = min(int(wrapped_y / self.resolution), self.rows - 1)
        return col, row

    def sample(self, position: Vec2) -> float:
        col, row = self.index_for_position(position)
        return self.values[(row * self.cols) + col]

    def consume_at_position(self, position: Vec2, amount: float) -> float:
        if amount <= 0.0:
            return 0.0

        col, row = self.index_for_position(position)
        index = (row * self.cols) + col
        consumed = min(amount, self.values[index])
        self.values[index] -= consumed
        return consumed

    def sample_gradient(self, position: Vec2) -> Vec2:
        step = self.resolution
        left = self.sample(Vec2(position.x - step, position.y))
        right = self.sample(Vec2(position.x + step, position.y))
        down = self.sample(Vec2(position.x, position.y - step))
        up = self.sample(Vec2(position.x, position.y + step))
        return Vec2(
            x=(right - left) / (2.0 * step),
            y=(up - down) / (2.0 * step),
        )

    def set_value(self, col: int, row: int, value: float) -> None:
        self.values[(row * self.cols) + col] = value

    def add_value(self, col: int, row: int, delta: float) -> None:
        index = (row * self.cols) + col
        self.values[index] += delta

    def add_value_capped(self, col: int, row: int, amount: float, cap: float) -> None:
        index = (row * self.cols) + col
        self.values[index] = min(cap, self.values[index] + amount)

    def add_value_at_position(self, position: Vec2, delta: float) -> None:
        col, row = self.index_for_position(position)
        self.add_value(col=col, row=row, delta=delta)

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
                projection = 0.5 + ((x - 0.5) * normalized[0]) + ((y - 0.5) * normalized[1])
                value = max(0.0, min(projection, 1.0)) * intensity
                self.set_value(col=col, row=row, value=value)

    def diffuse(self, rate: float) -> None:
        if rate <= 0.0:
            return

        original = list(self.values)
        updated = [0.0] * len(self.values)
        bounded = self.boundary == "bounded"
        for row in range(self.rows):
            for col in range(self.cols):
                index = (row * self.cols) + col
                center = original[index]
                if bounded:
                    left_col = max(0, col - 1)
                    right_col = min(self.cols - 1, col + 1)
                    up_row = max(0, row - 1)
                    down_row = min(self.rows - 1, row + 1)
                else:
                    left_col = (col - 1) % self.cols
                    right_col = (col + 1) % self.cols
                    up_row = (row - 1) % self.rows
                    down_row = (row + 1) % self.rows
                neighbor_total = (
                    original[(row * self.cols) + left_col]
                    + original[(row * self.cols) + right_col]
                    + original[(up_row * self.cols) + col]
                    + original[(down_row * self.cols) + col]
                )
                neighbor_average = neighbor_total / 4.0
                updated[index] = (center * (1.0 - rate)) + (neighbor_average * rate)
        self.values = updated

    def decay(self, rate: float) -> None:
        if rate <= 0.0:
            return

        factor = max(0.0, 1.0 - rate)
        self.values = [value * factor for value in self.values]
