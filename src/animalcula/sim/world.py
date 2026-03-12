"""Minimal world skeleton for the first testable physics slice."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import random
from typing import Callable

from animalcula.config import Config
from animalcula.sim.energy import basal_cost, feeding_gain, photosynthesis_gain
from animalcula.sim.fields import Grid2D
from animalcula.sim.physics import apply_edge_springs, apply_overdamped_dynamics
from animalcula.sim.seeding import build_demo_archetypes
from animalcula.sim.types import CreatureState, EdgeState, NodeState, NodeType


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
        creatures: list[CreatureState] | None = None,
    ) -> None:
        self.config = config
        self.seed = config.simulation.initial_seed if seed is None else seed
        self.tick = 0
        self.nodes = list(nodes or [])
        self.edges = list(edges or [])
        self.creatures = list(creatures or [])
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
        self._nutrient_source_cells = self._initialize_nutrient_sources()
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
            population=len(self.creatures) if self.creatures else len(self.nodes),
            phase_trace=list(self._phase_trace),
        )

    def random_unit(self) -> float:
        return self._rng.random()

    def seed_demo_archetypes(self) -> None:
        nodes, edges, creatures = build_demo_archetypes(
            world_width=self.config.world.width,
            world_height=self.config.world.height,
            nutrient_grid=self.nutrient_grid,
            nutrient_source_cells=self._nutrient_source_cells,
        )
        self.nodes.extend(nodes)
        self.edges.extend(edges)
        self.creatures.extend(creatures)

    def _step_once(self) -> Snapshot:
        self._phase_trace = []
        self._run_phase("environment", self._update_environment)
        self._run_phase("sensing", self._sense_environment)
        self._run_phase("brain", self._update_brains)
        self._run_phase("physics", self._apply_physics)
        self._run_phase("energy", self._apply_energy)
        self._run_phase("lifecycle", self._apply_lifecycle)
        self.tick += 1
        return self.snapshot()

    def _run_phase(self, name: str, func: Callable[[], None]) -> None:
        self._phase_trace.append(name)
        func()

    def _update_environment(self) -> None:
        for col, row in self._nutrient_source_cells:
            self.nutrient_grid.set_value(
                col=col,
                row=row,
                value=self.config.environment.nutrient_source_strength,
            )
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

    def _apply_energy(self) -> None:
        updated_creatures: list[CreatureState] = []
        for creature in self.creatures:
            creature_nodes = [self.nodes[index] for index in creature.node_indices]
            receptor_nodes = [
                node for node in creature_nodes if node.node_type == NodeType.PHOTORECEPTOR
            ]
            mouth_nodes = [node for node in creature_nodes if node.node_type == NodeType.MOUTH]
            if receptor_nodes:
                average_light = sum(
                    self.light_grid.sample(node.position) for node in receptor_nodes
                ) / len(receptor_nodes)
            else:
                average_light = 0.0
            if mouth_nodes:
                average_nutrients = sum(
                    self.nutrient_grid.sample(node.position) for node in mouth_nodes
                ) / len(mouth_nodes)
            else:
                average_nutrients = 0.0

            gain = photosynthesis_gain(
                light_level=average_light,
                receptor_count=len(receptor_nodes),
                photosynthesis_rate=self.config.energy.photosynthesis_rate,
            )
            gain += feeding_gain(
                nutrient_level=average_nutrients,
                mouth_count=len(mouth_nodes),
                feed_rate=self.config.energy.feed_rate,
            )
            cost = basal_cost(
                node_count=len(creature_nodes),
                basal_cost_per_node=self.config.energy.basal_cost_per_node,
            )
            updated_creatures.append(replace(creature, energy=creature.energy + gain - cost))

        self.creatures = updated_creatures

    def _apply_lifecycle(self) -> None:
        if not self.creatures:
            return None

        living_creatures = [creature for creature in self.creatures if creature.energy > 0.0]
        if len(living_creatures) == len(self.creatures):
            return None

        live_node_indices = sorted(
            {index for creature in living_creatures for index in creature.node_indices}
        )
        node_index_map = {old: new for new, old in enumerate(live_node_indices)}
        self.nodes = [self.nodes[index] for index in live_node_indices]
        self.edges = [
            EdgeState(
                a=node_index_map[edge.a],
                b=node_index_map[edge.b],
                rest_length=edge.rest_length,
                stiffness=edge.stiffness,
            )
            for edge in self.edges
            if edge.a in node_index_map and edge.b in node_index_map
        ]
        self.creatures = [
            replace(
                creature,
                node_indices=tuple(node_index_map[index] for index in creature.node_indices),
            )
            for creature in living_creatures
        ]
        return None

    def _initialize_nutrient_sources(self) -> list[tuple[int, int]]:
        source_cells: set[tuple[int, int]] = set()
        while len(source_cells) < self.config.environment.nutrient_source_count:
            source_cells.add(
                (
                    self._rng.randrange(self.nutrient_grid.cols),
                    self._rng.randrange(self.nutrient_grid.rows),
                )
            )
        return sorted(source_cells)
