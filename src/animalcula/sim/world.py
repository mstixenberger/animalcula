"""Minimal world skeleton for the first testable physics slice."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import json
from pathlib import Path
import random
from typing import Callable

from animalcula.config import Config
from animalcula.sim.brain import step_brain
from animalcula.sim.energy import basal_cost, photosynthesis_gain
from animalcula.sim.fields import Grid2D
from animalcula.sim.mutation import mutate_brain, mutate_edge, mutate_node
from animalcula.sim.physics import (
    apply_edge_springs,
    apply_motor_forces,
    apply_node_repulsion,
    apply_overdamped_dynamics,
)
from animalcula.sim.seeding import build_demo_archetypes
from animalcula.sim.types import BrainState, CreatureState, EdgeState, EventRecord, NodeState, NodeType, Vec2


@dataclass(slots=True, frozen=True)
class Snapshot:
    tick: int
    population: int
    phase_trace: list[str]


@dataclass(slots=True, frozen=True)
class Stats:
    tick: int
    population: int
    node_count: int
    edge_count: int
    total_energy: float
    births: int
    deaths: int
    reproductions: int


class World:
    """Headless world facade with deterministic seeded state."""

    def __init__(
        self,
        config: Config,
        seed: int | None = None,
        turbo: bool = False,
        nodes: list[NodeState] | None = None,
        edges: list[EdgeState] | None = None,
        creatures: list[CreatureState] | None = None,
    ) -> None:
        self.config = config
        self.seed = config.simulation.initial_seed if seed is None else seed
        self.turbo = turbo
        self.tick = 0
        self.nodes = list(nodes or [])
        self.edges = list(edges or [])
        self.creatures = list(creatures or [])
        self.events: list[EventRecord] = []
        self._phase_trace: list[str] = []
        self._rng = random.Random(self.seed)
        self._next_creature_id = self._initial_next_creature_id()
        self.creatures = self._assign_creature_ids(self.creatures)
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
        self.detritus_grid = Grid2D(
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

    def get_top_creatures(self, n: int, metric: str = "energy") -> list[CreatureState]:
        if metric != "energy":
            msg = f"unsupported creature ranking metric: {metric}"
            raise ValueError(msg)
        if n <= 0:
            return []
        return sorted(self.creatures, key=lambda creature: creature.energy, reverse=True)[:n]

    def snapshot(self) -> Snapshot:
        return Snapshot(
            tick=self.tick,
            population=len(self.creatures) if self.creatures else len(self.nodes),
            phase_trace=list(self._phase_trace),
        )

    def stats(self) -> Stats:
        births = sum(1 for event in self.events if event.event_type == "birth")
        deaths = sum(1 for event in self.events if event.event_type == "death")
        reproductions = sum(1 for event in self.events if event.event_type == "reproduction")
        return Stats(
            tick=self.tick,
            population=len(self.creatures) if self.creatures else len(self.nodes),
            node_count=len(self.nodes),
            edge_count=len(self.edges),
            total_energy=sum(creature.energy for creature in self.creatures),
            births=births,
            deaths=deaths,
            reproductions=reproductions,
        )

    def save(self, path: str | Path) -> None:
        payload = {
            "config": self.config.to_dict(),
            "seed": self.seed,
            "tick": self.tick,
            "nodes": [
                {
                    "position": [node.position.x, node.position.y],
                    "velocity": [node.velocity.x, node.velocity.y],
                    "accumulated_force": [
                        node.accumulated_force.x,
                        node.accumulated_force.y,
                    ],
                    "drag_coeff": node.drag_coeff,
                    "radius": node.radius,
                    "node_type": node.node_type.value,
                }
                for node in self.nodes
            ],
            "edges": [
                {
                    "a": edge.a,
                    "b": edge.b,
                    "rest_length": edge.rest_length,
                    "stiffness": edge.stiffness,
                    "has_motor": edge.has_motor,
                    "motor_strength": edge.motor_strength,
                }
                for edge in self.edges
            ],
            "creatures": [
                {
                    "node_indices": list(creature.node_indices),
                    "energy": creature.energy,
                    "id": creature.id,
                    "parent_id": creature.parent_id,
                    "age_ticks": creature.age_ticks,
                    "brain": None
                    if creature.brain is None
                    else {
                        "input_weights": [list(row) for row in creature.brain.input_weights],
                        "recurrent_weights": [list(row) for row in creature.brain.recurrent_weights],
                        "biases": list(creature.brain.biases),
                        "time_constants": list(creature.brain.time_constants),
                        "states": list(creature.brain.states),
                        "output_size": creature.brain.output_size,
                    },
                    "last_sensed_inputs": list(creature.last_sensed_inputs),
                    "last_brain_outputs": list(creature.last_brain_outputs),
                }
                for creature in self.creatures
            ],
            "events": [
                {
                    "tick": event.tick,
                    "event_type": event.event_type,
                    "creature_id": event.creature_id,
                    "parent_ids": list(event.parent_ids),
                    "energy": event.energy,
                }
                for event in self.events
            ],
            "nutrient_grid": self.nutrient_grid.values,
            "light_grid": self.light_grid.values,
            "detritus_grid": self.detritus_grid.values,
            "nutrient_source_cells": self._nutrient_source_cells,
            "next_creature_id": self._next_creature_id,
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "World":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        world = cls(
            config=Config.from_dict(payload["config"]),
            seed=payload["seed"],
            nodes=[
                NodeState(
                    position=Vec2(*node["position"]),
                    velocity=Vec2(*node["velocity"]),
                    accumulated_force=Vec2(*node["accumulated_force"]),
                    drag_coeff=node["drag_coeff"],
                    radius=node["radius"],
                    node_type=NodeType(node["node_type"]),
                )
                for node in payload["nodes"]
            ],
            edges=[
                EdgeState(
                    a=edge["a"],
                    b=edge["b"],
                    rest_length=edge["rest_length"],
                    stiffness=edge["stiffness"],
                    has_motor=edge.get("has_motor", False),
                    motor_strength=edge.get("motor_strength", 0.0),
                )
                for edge in payload["edges"]
            ],
            creatures=[
                CreatureState(
                    node_indices=tuple(creature["node_indices"]),
                    energy=creature["energy"],
                    brain=None
                    if creature["brain"] is None
                    else BrainState(
                        input_weights=tuple(tuple(row) for row in creature["brain"]["input_weights"]),
                        recurrent_weights=tuple(
                            tuple(row) for row in creature["brain"]["recurrent_weights"]
                        ),
                        biases=tuple(creature["brain"]["biases"]),
                        time_constants=tuple(creature["brain"]["time_constants"]),
                        states=tuple(creature["brain"]["states"]),
                        output_size=creature["brain"]["output_size"],
                    ),
                    last_sensed_inputs=tuple(creature.get("last_sensed_inputs", [])),
                    last_brain_outputs=tuple(creature.get("last_brain_outputs", [])),
                    id=creature.get("id", -1),
                    parent_id=creature.get("parent_id"),
                    age_ticks=creature.get("age_ticks", 0),
                )
                for creature in payload["creatures"]
            ],
        )
        world.tick = payload["tick"]
        world.events = [
            EventRecord(
                tick=event["tick"],
                event_type=event["event_type"],
                creature_id=event["creature_id"],
                parent_ids=tuple(event.get("parent_ids", [])),
                energy=event.get("energy", 0.0),
            )
            for event in payload.get("events", [])
        ]
        world.nutrient_grid.values = payload["nutrient_grid"]
        world.light_grid.values = payload["light_grid"]
        world.detritus_grid.values = payload.get("detritus_grid", [0.0] * len(world.detritus_grid.values))
        world._nutrient_source_cells = [tuple(cell) for cell in payload["nutrient_source_cells"]]
        world._next_creature_id = payload.get("next_creature_id", world._initial_next_creature_id())
        return world

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
        seeded_creatures = self._assign_creature_ids(creatures)
        self.creatures.extend(seeded_creatures)
        for creature in seeded_creatures:
            self._record_event("birth", creature_id=creature.id, parent_ids=(), energy=creature.energy)

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
        if (not self.turbo) or ((self.tick + 1) % 4 == 0):
            self.nutrient_grid.diffuse(rate=self.config.environment.nutrient_diffusion_rate)
            self.nutrient_grid.decay(rate=self.config.environment.nutrient_decay_rate)
            self._recycle_detritus()
            self.detritus_grid.decay(rate=self.config.environment.detritus_decay_rate)
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
        updated_creatures: list[CreatureState] = []
        for creature in self.creatures:
            creature = replace(creature, age_ticks=creature.age_ticks + 1)
            creature_nodes = [self.nodes[index] for index in creature.node_indices]
            receptor_nodes = [
                node for node in creature_nodes if node.node_type == NodeType.PHOTORECEPTOR
            ]
            mouth_nodes = [node for node in creature_nodes if node.node_type == NodeType.MOUTH]
            average_light = (
                sum(self.light_grid.sample(node.position) for node in receptor_nodes) / len(receptor_nodes)
                if receptor_nodes
                else 0.0
            )
            average_light_gradient = (
                sum(
                    (self.light_grid.sample_gradient(node.position) for node in receptor_nodes),
                    start=Vec2.zero(),
                )
                / len(receptor_nodes)
                if receptor_nodes
                else Vec2.zero()
            )
            average_nutrients = (
                sum(self.nutrient_grid.sample(node.position) for node in mouth_nodes) / len(mouth_nodes)
                if mouth_nodes
                else 0.0
            )
            average_nutrient_gradient = (
                sum(
                    (self.nutrient_grid.sample_gradient(node.position) for node in mouth_nodes),
                    start=Vec2.zero(),
                )
                / len(mouth_nodes)
                if mouth_nodes
                else Vec2.zero()
            )
            normalized_energy = min(
                1.0,
                creature.energy / max(self.config.energy.reproduction_threshold, 1.0),
            )
            updated_creatures.append(
                replace(
                    creature,
                    last_sensed_inputs=(
                        average_light,
                        average_nutrients,
                        normalized_energy,
                        average_light_gradient.x,
                        average_light_gradient.y,
                        average_nutrient_gradient.x,
                        average_nutrient_gradient.y,
                        min(1.0, creature.age_ticks / 1000.0),
                    ),
                )
            )
        self.creatures = updated_creatures

    def _update_brains(self) -> None:
        updated_creatures: list[CreatureState] = []
        for creature in self.creatures:
            if creature.brain is None:
                updated_creatures.append(creature)
                continue
            brain, outputs = step_brain(
                brain=creature.brain,
                inputs=creature.last_sensed_inputs,
                dt=self.config.physics.dt,
            )
            updated_creatures.append(
                replace(
                    creature,
                    brain=brain,
                    last_brain_outputs=outputs,
                )
            )
        self.creatures = updated_creatures

    def _apply_physics(self) -> None:
        edge_outputs: dict[int, float] = {}
        for creature in self.creatures:
            if not creature.last_brain_outputs:
                continue
            node_index_set = set(creature.node_indices)
            motor_edges = [
                edge_index
                for edge_index, edge in enumerate(self.edges)
                if edge.has_motor and edge.a in node_index_set and edge.b in node_index_set
            ]
            if motor_edges:
                for output, edge_index in zip(creature.last_brain_outputs, motor_edges, strict=False):
                    edge_outputs[edge_index] = (2.0 * output) - 1.0
            else:
                drive = (2.0 * creature.last_brain_outputs[0]) - 1.0
                force = Vec2(self.config.brain.motor_force_scale * drive, 0.0)
                per_node_force = force / max(len(creature.node_indices), 1)
                for node_index in creature.node_indices:
                    self.nodes[node_index] = replace(
                        self.nodes[node_index],
                        accumulated_force=self.nodes[node_index].accumulated_force + per_node_force,
                    )
        self.nodes = apply_motor_forces(nodes=self.nodes, edges=self.edges, edge_outputs=edge_outputs)
        self.nodes = apply_edge_springs(nodes=self.nodes, edges=self.edges)
        self.nodes = apply_node_repulsion(
            nodes=self.nodes,
            strength=self.config.physics.contact_repulsion,
        )
        self.nodes = [
            apply_overdamped_dynamics(node=node, dt=self.config.physics.dt)
            for node in self.nodes
        ]

    def _apply_energy(self) -> None:
        updated_creatures: list[CreatureState] = []
        population_count = len(self.creatures)
        crowding_multiplier = max(
            1.0,
            population_count / max(self.config.creatures.max_population, 1),
        )
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
            gain += sum(
                self.nutrient_grid.consume_at_position(
                    position=node.position,
                    amount=self.config.energy.feed_rate * self.nutrient_grid.sample(node.position),
                )
                for node in mouth_nodes
            )
            cost = basal_cost(
                node_count=len(creature_nodes),
                basal_cost_per_node=self.config.energy.basal_cost_per_node,
            )
            updated_creatures.append(
                replace(
                    creature,
                    energy=creature.energy + gain - (cost * crowding_multiplier),
                )
            )

        self.creatures = updated_creatures

    def _apply_lifecycle(self) -> None:
        if not self.creatures:
            return None

        self._reproduce_creatures()
        dead_creatures = [creature for creature in self.creatures if creature.energy <= 0.0]
        for creature in dead_creatures:
            self._deposit_detritus(creature)
            self._record_event("death", creature_id=creature.id, parent_ids=(), energy=creature.energy)
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
                has_motor=edge.has_motor,
                motor_strength=edge.motor_strength,
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
        self._apply_population_floor()
        return None

    def _reproduce_creatures(self) -> None:
        new_nodes: list[NodeState] = []
        new_edges: list[EdgeState] = []
        new_creatures: list[CreatureState] = []
        updated_creatures: list[CreatureState] = []

        for creature_index, creature in enumerate(self.creatures):
            if creature.energy < self.config.energy.reproduction_threshold:
                updated_creatures.append(creature)
                continue

            child_offset = Vec2(2.0 * (creature_index + 1), 2.0 * (creature_index + 1))
            node_index_map: dict[int, int] = {}
            for node_index in creature.node_indices:
                cloned = mutate_node(
                    node=replace(
                        self.nodes[node_index],
                        position=self.nodes[node_index].position + child_offset,
                        velocity=Vec2.zero(),
                        accumulated_force=Vec2.zero(),
                    ),
                    rng=self._rng,
                    position_sigma=self.config.evolution.position_mutation_sigma,
                    radius_sigma=self.config.evolution.radius_mutation_sigma,
                )
                node_index_map[node_index] = len(self.nodes) + len(new_nodes)
                new_nodes.append(cloned)

            for edge in self.edges:
                if edge.a in node_index_map and edge.b in node_index_map:
                    new_edges.append(
                        mutate_edge(
                            edge=EdgeState(
                                a=node_index_map[edge.a],
                                b=node_index_map[edge.b],
                                rest_length=edge.rest_length,
                                stiffness=edge.stiffness,
                                has_motor=edge.has_motor,
                                motor_strength=edge.motor_strength,
                            ),
                            rng=self._rng,
                            motor_strength_sigma=self.config.evolution.motor_strength_mutation_sigma,
                        )
                    )

            split_energy = creature.energy / 2.0
            updated_creatures.append(replace(creature, energy=split_energy))
            child_brain = (
                None
                if creature.brain is None
                else mutate_brain(
                    brain=creature.brain,
                    rng=self._rng,
                    weight_sigma=self.config.evolution.weight_mutation_sigma,
                    bias_sigma=self.config.evolution.bias_mutation_sigma,
                    tau_sigma=self.config.evolution.tau_mutation_sigma,
                )
            )
            child_id = self._allocate_creature_id()
            new_creatures.append(
                CreatureState(
                    node_indices=tuple(node_index_map[index] for index in creature.node_indices),
                    energy=split_energy,
                    brain=child_brain,
                    id=child_id,
                    parent_id=creature.id,
                )
            )
            self._record_event(
                "reproduction",
                creature_id=creature.id,
                parent_ids=(creature.id,),
                energy=split_energy,
            )
            self._record_event(
                "birth",
                creature_id=child_id,
                parent_ids=(creature.id,),
                energy=split_energy,
            )

        self.nodes.extend(new_nodes)
        self.edges.extend(new_edges)
        self.creatures = updated_creatures + new_creatures

    def _apply_population_floor(self) -> None:
        while len(self.creatures) < self.config.creatures.min_population:
            self.seed_demo_archetypes()

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

    def _initial_next_creature_id(self) -> int:
        existing_ids = [creature.id for creature in self.creatures if creature.id >= 0]
        return (max(existing_ids) + 1) if existing_ids else 1

    def _allocate_creature_id(self) -> int:
        creature_id = self._next_creature_id
        self._next_creature_id += 1
        return creature_id

    def _assign_creature_ids(self, creatures: list[CreatureState]) -> list[CreatureState]:
        assigned: list[CreatureState] = []
        seen_ids: set[int] = set()
        for creature in creatures:
            creature_id = creature.id
            if creature_id < 0 or creature_id in seen_ids:
                creature_id = self._allocate_creature_id()
            seen_ids.add(creature_id)
            assigned.append(replace(creature, id=creature_id))
        return assigned

    def _record_event(
        self,
        event_type: str,
        creature_id: int,
        parent_ids: tuple[int, ...],
        energy: float,
    ) -> None:
        self.events.append(
            EventRecord(
                tick=self.tick,
                event_type=event_type,
                creature_id=creature_id,
                parent_ids=parent_ids,
                energy=energy,
            )
        )

    def _deposit_detritus(self, creature: CreatureState) -> None:
        creature_nodes = [self.nodes[index] for index in creature.node_indices]
        if not creature_nodes:
            return

        for node in creature_nodes:
            self.detritus_grid.add_value_at_position(position=node.position, delta=max(node.radius, 0.1))

    def _recycle_detritus(self) -> None:
        rate = self.config.environment.detritus_recycling_rate
        if rate <= 0.0:
            return

        recycled_values: list[float] = []
        for index, detritus in enumerate(self.detritus_grid.values):
            recycled = detritus * rate
            self.nutrient_grid.values[index] += recycled
            recycled_values.append(detritus - recycled)
        self.detritus_grid.values = recycled_values
