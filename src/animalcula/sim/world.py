"""Minimal world skeleton for the first testable slice."""

from __future__ import annotations

from dataclasses import dataclass
import random

from animalcula.config import Config


@dataclass(slots=True, frozen=True)
class Snapshot:
    tick: int
    population: int


class World:
    """Headless world facade with deterministic seeded state."""

    def __init__(self, config: Config, seed: int | None = None) -> None:
        self.config = config
        self.seed = config.simulation.initial_seed if seed is None else seed
        self.tick = 0
        self._rng = random.Random(self.seed)

    def step(self, ticks: int = 1) -> Snapshot:
        if ticks < 0:
            raise ValueError("ticks must be non-negative")

        self.tick += ticks
        return self.snapshot()

    def snapshot(self) -> Snapshot:
        return Snapshot(tick=self.tick, population=0)

    def random_unit(self) -> float:
        return self._rng.random()
