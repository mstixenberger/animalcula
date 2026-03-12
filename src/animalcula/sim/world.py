"""Minimal world skeleton for the first testable physics slice."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable

from animalcula.config import Config
from animalcula.sim.fields import Grid2D
from animalcula.sim.physics import apply_edge_springs, apply_overdamped_dynamics
from animalcula.sim.types import EdgeState, NodeState


@dataclass(slots=True, frozen=True)
class Snapshot:
    tick: int
    population: int
    phase_trace: list[str]


class World:
    """Headless world facade with deterministic seeded state."""

    def __init__(
        self,
        config: Config,
        seed: int | None = None,
        nodes: list[NodeState] | None = None,
        edges: list[EdgeState] | None = None,
    ) -> None:
        self.config = config
        self.seed = config.simulation.initial_seed if seed is None else seed
        self.tick = 0
        self.nodes = list(nodes or [])
        self.edges = list(edges or [])
        self._phase_trace: list[str] = []
        self._rng = random.Random(self.seed)
        self.nutrient_grid = Grid2D(
            width=self.config.world.width,
            height=self.config.world.height,
            resolution=self.config.world.grid_resolution,
        )
        self.light_grid = Grid2D(
            width=self.config.world.width,
            height=self.config.world.height,
            resolution=self.config.world.grid_resolution,
        )
        self._update_environment()

    def step(self, ticks: int = 1) -> Snapshot:
        if ticks < 0:
            raise ValueError("ticks must be non-negative")

        snapshot = self.snapshot()
        for _ in range(ticks):
            snapshot = self._step_once()
        return snapshot

    def snapshot(self) -> Snapshot:
        return Snapshot(
            tick=self.tick,
            population=len(self.nodes),
            phase_trace=list(self._phase_trace),
        )

    def random_unit(self) -> float:
        return self._rng.random()

    def _step_once(self) -> Snapshot:
        self._phase_trace = []
        self._run_phase("environment", self._update_environment)
        self._run_phase("sensing", self._sense_environment)
        self._run_phase("brain", self._update_brains)
        self._run_phase("physics", self._apply_physics)
        self._run_phase("lifecycle", self._apply_lifecycle)
        self.tick += 1
        return self.snapshot()

    def _run_phase(self, name: str, func: Callable[[], None]) -> None:
        self._phase_trace.append(name)
        func()

    def _update_environment(self) -> None:
        self.light_grid.fill_light_gradient(
            direction=self.config.environment.light_direction,
            intensity=self.config.environment.light_intensity_max,
        )

    def _sense_environment(self) -> None:
        return None

    def _update_brains(self) -> None:
        return None

    def _apply_physics(self) -> None:
        self.nodes = apply_edge_springs(nodes=self.nodes, edges=self.edges)
        self.nodes = [
            apply_overdamped_dynamics(node=node, dt=self.config.physics.dt)
            for node in self.nodes
        ]

    def _apply_lifecycle(self) -> None:
        return None
