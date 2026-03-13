"""Minimal world skeleton for the first testable physics slice."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import json
import math
from pathlib import Path
import random
from collections import Counter
from typing import Callable

from animalcula.analysis.metrics import shannon_diversity
from animalcula.config import Config
from animalcula.sim.brain import step_brain
from animalcula.sim.energy import basal_cost, motor_cost, photosynthesis_gain
from animalcula.sim.fields import Grid2D
from animalcula.sim.genome import (
    cluster_species,
    coarse_species_signature,
    decode_genome,
    encode_creature_genome,
    genome_from_dict,
    genome_hash,
    genome_to_dict,
    mutate_genome,
)
from animalcula.sim.physics import (
    apply_edge_springs,
    apply_grip_latches,
    apply_motor_forces,
    apply_node_repulsion,
    apply_overdamped_dynamics,
    spring_force,
)
from animalcula.sim.seeding import build_demo_archetypes
from animalcula.sim.types import BrainState, CreatureState, EdgeState, EventRecord, GripLatch, NodeState, NodeType, Vec2


@dataclass(slots=True, frozen=True)
class Snapshot:
    tick: int
    population: int
    phase_trace: list[str]
    world_width: float
    world_height: float
    total_energy: float
    nodes: tuple["NodeSnapshot", ...]
    edges: tuple["EdgeSnapshot", ...]
    creatures: tuple["CreatureSnapshot", ...]


@dataclass(slots=True, frozen=True)
class NodeSnapshot:
    x: float
    y: float
    radius: float
    node_type: str
    creature_id: int | None


@dataclass(slots=True, frozen=True)
class EdgeSnapshot:
    ax: float
    ay: float
    bx: float
    by: float
    has_motor: bool


@dataclass(slots=True, frozen=True)
class CreatureSnapshot:
    creature_id: int
    energy: float
    trophic_role: str
    center_x: float
    center_y: float


@dataclass(slots=True, frozen=True)
class Stats:
    tick: int
    population: int
    peak_population: int
    population_variance: float
    population_capacity_fraction: float
    peak_population_capacity_fraction: float
    crowding_multiplier: float
    peak_crowding_multiplier: float
    node_count: int
    edge_count: int
    total_energy: float
    births: int
    deaths: int
    reproductions: int
    speciation_events: int
    species_extinctions: int
    species_turnover: int
    predation_kills: int
    environment_perturbations: int
    lineage_count: int
    species_count: int
    observed_species_count: int
    peak_species_count: int
    diversity_index: float
    mean_nodes_per_creature: float
    longest_species_lifespan: int
    mean_extinct_species_lifespan: float
    autotroph_count: int
    herbivore_count: int
    predator_count: int
    drag_multiplier: float
    nutrient_source_strength_multiplier: float
    peak_species_fraction: float
    runaway_dominance_detected: bool


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
        self.grip_latches: list[GripLatch] = []
        self._predation_kill_ids: set[int] = set()
        self._phase_trace: list[str] = []
        self._rng = random.Random(self.seed)
        self._next_creature_id = self._initial_next_creature_id()
        self.creatures = self._assign_creature_ids(self.creatures)
        self.creatures = self._ensure_creature_genomes(self.creatures)
        self._known_species_ids = set(self._species_labels().values())
        self._species_first_seen_tick = {species_id: self.tick for species_id in self._known_species_ids}
        self._species_last_seen_tick = {species_id: self.tick for species_id in self._known_species_ids}
        self._extinct_species_ids: set[str] = set()
        self._peak_species_count = len(self._known_species_ids)
        self._peak_species_fraction = 0.0
        self._runaway_dominance_detected = False
        self._dominant_species_id: str | None = None
        self._dominant_species_streak = 0
        self._runaway_dominance_fraction_threshold = 0.8
        self._runaway_dominance_tick_threshold = 5000
        self._population_observation_count = 0
        self._population_mean = 0.0
        self._population_m2 = 0.0
        self._peak_population = 0
        self._update_species_dominance_metrics(self._species_labels())
        self._record_population_observation()
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
        self.chemical_a_grid = Grid2D(
            width=self.config.world.width,
            height=self.config.world.height,
            resolution=self.config.world.grid_resolution,
        )
        self.chemical_b_grid = Grid2D(
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

    def export_top_creatures(self, path: str | Path, n: int, metric: str = "energy") -> None:
        payload = self.top_creature_payload(n=n, metric=metric)
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def top_creature_payload(self, n: int, metric: str = "energy") -> list[dict[str, object]]:
        return [self._serialize_creature(creature) for creature in self.get_top_creatures(n=n, metric=metric)]

    def species_snapshots(self) -> list[dict[str, float | int | str]]:
        species_labels = self._species_labels()
        grouped: dict[str, list[CreatureState]] = {}
        for creature in self.creatures:
            species_id = species_labels.get(creature.id, coarse_species_signature(creature.genome))
            grouped.setdefault(species_id, []).append(creature)

        snapshots: list[dict[str, float | int | str]] = []
        for species_id, creatures in sorted(grouped.items()):
            node_counts = [len(creature.node_indices) for creature in creatures]
            snapshots.append(
                {
                    "tick": self.tick,
                    "species_id": species_id,
                    "count": len(creatures),
                    "mean_energy": sum(creature.energy for creature in creatures) / len(creatures),
                    "mean_size": sum(node_counts) / len(node_counts),
                    "mean_speed": sum(creature.mean_speed_recent for creature in creatures) / len(creatures),
                }
            )
        return snapshots

    def phenotype_snapshots(self) -> list[dict[str, float | int | str]]:
        species_labels = self._species_labels()
        snapshots: list[dict[str, float | int | str]] = []
        for creature in self.creatures:
            node_states = [self.nodes[node_index] for node_index in creature.node_indices]
            mouths = sum(1 for node in node_states if node.node_type == NodeType.MOUTH)
            grippers = sum(1 for node in node_states if node.node_type == NodeType.GRIPPER)
            sensors = sum(1 for node in node_states if node.node_type == NodeType.SENSOR)
            photoreceptors = sum(1 for node in node_states if node.node_type == NodeType.PHOTORECEPTOR)
            creature_node_set = set(creature.node_indices)
            creature_edges = [
                edge for edge in self.edges if edge.a in creature_node_set and edge.b in creature_node_set
            ]
            edge_count = len(creature_edges)
            snapshots.append(
                {
                    "tick": self.tick,
                    "creature_id": creature.id,
                    "species_id": species_labels.get(creature.id, coarse_species_signature(creature.genome)),
                    "num_nodes": len(creature.node_indices),
                    "num_edges": edge_count,
                    "mean_segment_length": (
                        sum(edge.rest_length for edge in creature_edges) / edge_count if edge_count else 0.0
                    ),
                    "num_mouths": mouths,
                    "num_grippers": grippers,
                    "num_sensors": sensors,
                    "num_photoreceptors": photoreceptors,
                    "energy": creature.energy,
                    "mean_speed_recent": creature.mean_speed_recent,
                }
            )
        return snapshots

    def seed_from_exported_genomes(self, path: str | Path) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise TypeError("exported genome file must contain a list of creatures")
        self.seed_from_exported_payload(payload)

    def seed_from_exported_payload(self, payload: list[dict[str, object]]) -> None:
        imported: list[CreatureState] = []
        for item in payload:
            if not isinstance(item, dict):
                raise TypeError("exported creature entries must be mappings")
            genome = genome_from_dict(item.get("genome"))
            if genome is None:
                continue
            anchor = Vec2(
                self._rng.uniform(0.0, self.config.world.width),
                self._rng.uniform(0.0, self.config.world.height),
            )
            decoded_nodes, decoded_edges, brain = decode_genome(
                genome=genome,
                anchor_position=anchor,
                drag_coeff=self.config.physics.default_drag,
            )
            node_start = len(self.nodes)
            node_indices = tuple(range(node_start, node_start + len(decoded_nodes)))
            self.nodes.extend(decoded_nodes)
            self.edges.extend(
                EdgeState(
                    a=node_indices[edge.a],
                    b=node_indices[edge.b],
                    rest_length=edge.rest_length,
                    stiffness=edge.stiffness,
                    has_motor=edge.has_motor,
                    motor_strength=edge.motor_strength,
                )
                for edge in decoded_edges
            )
            imported.append(
                CreatureState(
                    node_indices=node_indices,
                    energy=float(item.get("energy", 1.0)),
                    brain=brain,
                    genome=genome,
                    age_ticks=0,
                )
            )

        seeded_creatures = self._ensure_creature_genomes(self._assign_creature_ids(imported))
        self.creatures.extend(seeded_creatures)
        self._sync_species_presence()
        for creature in seeded_creatures:
            self._record_event(
                "birth",
                creature_id=creature.id,
                parent_ids=(),
                energy=creature.energy,
                genome_hash_value=genome_hash(creature.genome),
            )

    def snapshot(self) -> Snapshot:
        node_to_creature = {
            node_index: creature.id
            for creature in self.creatures
            for node_index in creature.node_indices
        }
        node_snapshots = tuple(
            NodeSnapshot(
                x=node.position.x,
                y=node.position.y,
                radius=node.radius,
                node_type=node.node_type.value,
                creature_id=node_to_creature.get(node_index),
            )
            for node_index, node in enumerate(self.nodes)
        )
        edge_snapshots = tuple(
            EdgeSnapshot(
                ax=self.nodes[edge.a].position.x,
                ay=self.nodes[edge.a].position.y,
                bx=self.nodes[edge.b].position.x,
                by=self.nodes[edge.b].position.y,
                has_motor=edge.has_motor,
            )
            for edge in self.edges
        )
        creature_snapshots = tuple(
            CreatureSnapshot(
                creature_id=creature.id,
                energy=creature.energy,
                trophic_role=self._trophic_role(creature),
                center_x=self._creature_centroid(creature).x if self._creature_centroid(creature) is not None else 0.0,
                center_y=self._creature_centroid(creature).y if self._creature_centroid(creature) is not None else 0.0,
            )
            for creature in self.creatures
        )
        return Snapshot(
            tick=self.tick,
            population=len(self.creatures) if self.creatures else len(self.nodes),
            phase_trace=list(self._phase_trace),
            world_width=self.config.world.width,
            world_height=self.config.world.height,
            total_energy=sum(creature.energy for creature in self.creatures),
            nodes=node_snapshots,
            edges=edge_snapshots,
            creatures=creature_snapshots,
        )

    def stats(self) -> Stats:
        births = sum(1 for event in self.events if event.event_type == "birth")
        deaths = sum(1 for event in self.events if event.event_type == "death")
        reproductions = sum(1 for event in self.events if event.event_type == "reproduction")
        speciation_events = sum(1 for event in self.events if event.event_type == "speciation")
        species_extinctions = sum(1 for event in self.events if event.event_type == "species_extinction")
        predation_kills = sum(1 for event in self.events if event.event_type == "predation_kill")
        environment_perturbations = sum(1 for event in self.events if event.event_type == "environment_perturbation")
        lineage_counts = Counter(
            genome_hash(creature.genome) for creature in self.creatures if creature.genome is not None
        )
        species_counts = Counter(self._species_labels().values())
        autotroph_count = 0
        herbivore_count = 0
        predator_count = 0
        for creature in self.creatures:
            trophic_role = self._trophic_role(creature)
            if trophic_role == "autotroph":
                autotroph_count += 1
            elif trophic_role == "herbivore":
                herbivore_count += 1
            elif trophic_role == "predator":
                predator_count += 1
        return Stats(
            tick=self.tick,
            population=len(self.creatures) if self.creatures else len(self.nodes),
            peak_population=self._peak_population,
            population_variance=self.population_variance(),
            population_capacity_fraction=self.population_capacity_fraction(),
            peak_population_capacity_fraction=self.peak_population_capacity_fraction(),
            crowding_multiplier=self.crowding_multiplier(),
            peak_crowding_multiplier=self.peak_crowding_multiplier(),
            node_count=len(self.nodes),
            edge_count=len(self.edges),
            total_energy=sum(creature.energy for creature in self.creatures),
            births=births,
            deaths=deaths,
            reproductions=reproductions,
            speciation_events=speciation_events,
            species_extinctions=species_extinctions,
            species_turnover=speciation_events + species_extinctions,
            predation_kills=predation_kills,
            environment_perturbations=environment_perturbations,
            lineage_count=len(lineage_counts),
            species_count=len(species_counts),
            observed_species_count=len(self._known_species_ids),
            peak_species_count=self._peak_species_count,
            diversity_index=shannon_diversity(dict(lineage_counts)),
            mean_nodes_per_creature=(len(self.nodes) / len(self.creatures)) if self.creatures else 0.0,
            longest_species_lifespan=self._longest_species_lifespan(),
            mean_extinct_species_lifespan=self._mean_extinct_species_lifespan(),
            autotroph_count=autotroph_count,
            herbivore_count=herbivore_count,
            predator_count=predator_count,
            drag_multiplier=self.current_drag_multiplier(),
            nutrient_source_strength_multiplier=self.current_nutrient_source_strength_multiplier(),
            peak_species_fraction=self._peak_species_fraction,
            runaway_dominance_detected=self._runaway_dominance_detected,
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
                    **self._serialize_creature(creature),
                    "node_indices": list(creature.node_indices),
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
                    "genome_hash": event.genome_hash,
                }
                for event in self.events
            ],
            "nutrient_grid": self.nutrient_grid.values,
            "light_grid": self.light_grid.values,
            "chemical_a_grid": self.chemical_a_grid.values,
            "chemical_b_grid": self.chemical_b_grid.values,
            "detritus_grid": self.detritus_grid.values,
            "nutrient_source_cells": self._nutrient_source_cells,
            "next_creature_id": self._next_creature_id,
            "known_species_ids": sorted(self._known_species_ids),
            "species_first_seen_tick": self._species_first_seen_tick,
            "species_last_seen_tick": self._species_last_seen_tick,
            "extinct_species_ids": sorted(self._extinct_species_ids),
            "peak_species_count": self._peak_species_count,
            "peak_species_fraction": self._peak_species_fraction,
            "runaway_dominance_detected": self._runaway_dominance_detected,
            "dominant_species_id": self._dominant_species_id,
            "dominant_species_streak": self._dominant_species_streak,
            "population_observation_count": self._population_observation_count,
            "population_mean": self._population_mean,
            "population_m2": self._population_m2,
            "peak_population": self._peak_population,
            "grip_latches": [
                {
                    "creature_a_id": latch.creature_a_id,
                    "node_a_index": latch.node_a_index,
                    "creature_b_id": latch.creature_b_id,
                    "node_b_index": latch.node_b_index,
                    "rest_length": latch.rest_length,
                }
                for latch in self.grip_latches
            ],
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
                    mean_speed_recent=creature.get("mean_speed_recent", 0.0),
                    id=creature.get("id", -1),
                    parent_id=creature.get("parent_id"),
                    age_ticks=creature.get("age_ticks", 0),
                    genome=genome_from_dict(creature.get("genome")),
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
                genome_hash=event.get("genome_hash", ""),
            )
            for event in payload.get("events", [])
        ]
        world.nutrient_grid.values = payload["nutrient_grid"]
        world.light_grid.values = payload["light_grid"]
        world.chemical_a_grid.values = payload.get(
            "chemical_a_grid",
            [0.0] * len(world.chemical_a_grid.values),
        )
        world.chemical_b_grid.values = payload.get(
            "chemical_b_grid",
            [0.0] * len(world.chemical_b_grid.values),
        )
        world.detritus_grid.values = payload.get("detritus_grid", [0.0] * len(world.detritus_grid.values))
        world._nutrient_source_cells = [tuple(cell) for cell in payload["nutrient_source_cells"]]
        world._next_creature_id = payload.get("next_creature_id", world._initial_next_creature_id())
        world._known_species_ids = set(payload.get("known_species_ids", world._species_labels().values()))
        world._species_first_seen_tick = dict(payload.get("species_first_seen_tick", {}))
        world._species_last_seen_tick = dict(payload.get("species_last_seen_tick", {}))
        world._extinct_species_ids = set(payload.get("extinct_species_ids", []))
        world._peak_species_count = payload.get("peak_species_count", len(world._species_labels()))
        world._peak_species_fraction = payload.get("peak_species_fraction", 0.0)
        world._runaway_dominance_detected = payload.get("runaway_dominance_detected", False)
        world._dominant_species_id = payload.get("dominant_species_id")
        world._dominant_species_streak = payload.get("dominant_species_streak", 0)
        world._population_observation_count = payload.get("population_observation_count", 1)
        world._population_mean = payload.get("population_mean", float(len(world.creatures)))
        world._population_m2 = payload.get("population_m2", 0.0)
        world._peak_population = payload.get("peak_population", len(world.creatures))
        world.grip_latches = [
            GripLatch(
                creature_a_id=latch["creature_a_id"],
                node_a_index=latch["node_a_index"],
                creature_b_id=latch["creature_b_id"],
                node_b_index=latch["node_b_index"],
                rest_length=latch["rest_length"],
            )
            for latch in payload.get("grip_latches", [])
        ]
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
        seeded_creatures = self._ensure_creature_genomes(self._assign_creature_ids(creatures))
        self.creatures.extend(seeded_creatures)
        self._sync_species_presence()
        for creature in seeded_creatures:
            self._record_event(
                "birth",
                creature_id=creature.id,
                parent_ids=(),
                energy=creature.energy,
                genome_hash_value=genome_hash(creature.genome),
            )

    def _step_once(self) -> Snapshot:
        self._phase_trace = []
        self._run_phase("environment", self._update_environment)
        self._run_phase("sensing", self._sense_environment)
        self._run_phase("brain", self._update_brains)
        self._run_phase("physics", self._apply_physics)
        self._run_phase("energy", self._apply_energy)
        self._run_phase("lifecycle", self._apply_lifecycle)
        self._record_population_observation()
        self.tick += 1
        return self.snapshot()

    def _run_phase(self, name: str, func: Callable[[], None]) -> None:
        self._phase_trace.append(name)
        func()

    def _update_environment(self) -> None:
        if (not self.turbo) or ((self.tick + 1) % 4 == 0):
            self.nutrient_grid.diffuse(rate=self.config.environment.nutrient_diffusion_rate)
            self.nutrient_grid.decay(rate=self.config.environment.nutrient_decay_rate)
            self.chemical_a_grid.diffuse(rate=self.config.environment.chemical_diffusion_rate)
            self.chemical_a_grid.decay(rate=self.config.environment.chemical_decay_rate)
            self.chemical_b_grid.diffuse(rate=self.config.environment.chemical_diffusion_rate)
            self.chemical_b_grid.decay(rate=self.config.environment.chemical_decay_rate)
            self._recycle_detritus()
            self.detritus_grid.decay(rate=self.config.environment.detritus_decay_rate)
        if not self._reseed_nutrient_sources_if_due():
            self._shift_nutrient_sources_if_due()
        self._apply_dominance_perturbation_if_due()
        nutrient_source_strength = (
            self.config.environment.nutrient_source_strength * self.current_nutrient_source_strength_multiplier()
        )
        for col, row in self._nutrient_source_cells:
            self.nutrient_grid.set_value(
                col=col,
                row=row,
                value=nutrient_source_strength,
            )
        light_direction, light_intensity = self.current_light_state()
        self.light_grid.fill_light_gradient(
            direction=light_direction,
            intensity=light_intensity,
        )

    def current_light_state(self) -> tuple[tuple[float, float], float]:
        base_direction = self.config.environment.light_direction
        max_intensity = self.config.environment.light_intensity_max
        min_intensity = max(0.0, min(self.config.environment.light_intensity_min, max_intensity))
        interval = self.config.environment.light_season_interval
        steps = max(1, self.config.environment.light_season_steps)

        if interval <= 0 or steps <= 1:
            return base_direction, max_intensity

        season_index = ((self.tick + 1) // interval) % steps
        phase = season_index / steps
        angle = 2.0 * math.pi * phase
        direction_x = (base_direction[0] * math.cos(angle)) - (base_direction[1] * math.sin(angle))
        direction_y = (base_direction[0] * math.sin(angle)) + (base_direction[1] * math.cos(angle))
        intensity = min_intensity + ((max_intensity - min_intensity) * (0.5 + (0.5 * math.cos(angle))))
        return (direction_x, direction_y), intensity

    def current_drag_multiplier(self) -> float:
        interval = self.config.environment.drag_shift_interval
        multipliers = self.config.environment.drag_shift_multipliers
        if interval <= 0 or len(multipliers) <= 1:
            return float(multipliers[0]) if multipliers else 1.0
        regime_index = ((self.tick + 1) // interval) % len(multipliers)
        return max(0.001, float(multipliers[regime_index]))

    def current_nutrient_source_strength_multiplier(self) -> float:
        interval = self.config.environment.nutrient_epoch_interval
        multipliers = self.config.environment.nutrient_epoch_strength_multipliers
        if interval <= 0 or len(multipliers) <= 1:
            return float(multipliers[0]) if multipliers else 1.0
        regime_index = ((self.tick + 1) // interval) % len(multipliers)
        return max(0.0, float(multipliers[regime_index]))

    def _sense_environment(self) -> None:
        updated_creatures: list[CreatureState] = []
        for creature in self.creatures:
            creature = replace(creature, age_ticks=creature.age_ticks + 1)
            creature_nodes = [self.nodes[index] for index in creature.node_indices]
            receptor_nodes = [
                node for node in creature_nodes if node.node_type == NodeType.PHOTORECEPTOR
            ]
            mouth_nodes = [node for node in creature_nodes if node.node_type == NodeType.MOUTH]
            gripper_nodes = [node for node in creature_nodes if node.node_type == NodeType.GRIPPER]
            sensor_nodes = [node for node in creature_nodes if node.node_type == NodeType.SENSOR]
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
            average_chemical_a = (
                sum(self.chemical_a_grid.sample(node.position) for node in sensor_nodes) / len(sensor_nodes)
                if sensor_nodes
                else 0.0
            )
            average_chemical_b = (
                sum(self.chemical_b_grid.sample(node.position) for node in sensor_nodes) / len(sensor_nodes)
                if sensor_nodes
                else 0.0
            )
            average_chemical_a_gradient = (
                sum(
                    (self.chemical_a_grid.sample_gradient(node.position) for node in sensor_nodes),
                    start=Vec2.zero(),
                )
                / len(sensor_nodes)
                if sensor_nodes
                else Vec2.zero()
            )
            average_chemical_b_gradient = (
                sum(
                    (self.chemical_b_grid.sample_gradient(node.position) for node in sensor_nodes),
                    start=Vec2.zero(),
                )
                / len(sensor_nodes)
                if sensor_nodes
                else Vec2.zero()
            )
            normalized_energy = min(
                1.0,
                creature.energy / max(self.config.energy.reproduction_threshold, 1.0),
            )
            grip_contact = self._gripper_contact_signal(creature)
            grip_active = self._grip_active_signal(creature)
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
                        average_chemical_a,
                        average_chemical_b,
                        average_chemical_a_gradient.x,
                        average_chemical_a_gradient.y,
                        average_chemical_b_gradient.x,
                        average_chemical_b_gradient.y,
                        grip_contact,
                        grip_active,
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
            motor_edges = self._motor_edge_indices_for_creature(creature)
            if motor_edges:
                motor_outputs, _, _, _, _ = self._control_outputs_for_creature(creature)
                for output, edge_index in zip(motor_outputs, motor_edges, strict=False):
                    edge_outputs[edge_index] = (2.0 * output) - 1.0
            elif not self._mouth_nodes_for_creature(creature):
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
        self._refresh_grip_latches()
        self.nodes = apply_grip_latches(
            nodes=self.nodes,
            latches=self.grip_latches,
            stiffness=self.config.physics.grip_spring_stiffness,
        )
        drag_multiplier = self.current_drag_multiplier()
        self.nodes = [
            apply_overdamped_dynamics(
                node=node,
                dt=self.config.physics.dt,
                drag_multiplier=drag_multiplier,
            )
            for node in self.nodes
        ]
        self.creatures = self._update_recent_speeds(self.creatures)

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
            gain += sum(
                self.detritus_grid.consume_at_position(
                    position=node.position,
                    amount=self.config.energy.scavenging_rate * self.detritus_grid.sample(node.position),
                )
                for node in mouth_nodes
            )
            cost = basal_cost(
                node_count=len(creature_nodes),
                basal_cost_per_node=self.config.energy.basal_cost_per_node,
            )
            if creature.last_brain_outputs:
                motor_edge_count = sum(
                    1
                    for edge in self.edges
                    if edge.has_motor and edge.a in creature.node_indices and edge.b in creature.node_indices
                )
                actuation = 0.0
                for output, edge in zip(
                    creature.last_brain_outputs[:motor_edge_count],
                    (
                        edge
                        for edge in self.edges
                        if edge.has_motor and edge.a in creature.node_indices and edge.b in creature.node_indices
                    ),
                    strict=False,
                ):
                    actuation += abs((2.0 * output) - 1.0) * edge.motor_strength
                if motor_edge_count == 0:
                    actuation += abs((2.0 * creature.last_brain_outputs[0]) - 1.0) * self.config.brain.motor_force_scale
                cost += motor_cost(
                    total_actuation=actuation,
                    motor_cost_per_unit=self.config.energy.motor_cost_per_unit,
                )
            cost += self.config.energy.grip_cost * self._active_grip_count(creature)
            updated_creatures.append(
                replace(
                    creature,
                    energy=creature.energy + gain - (cost * crowding_multiplier),
                )
            )

        self._emit_chemicals(updated_creatures)
        updated_creatures = self._apply_predation(updated_creatures)
        self.creatures = updated_creatures

    def _apply_lifecycle(self) -> None:
        if not self.creatures:
            return None

        self._reproduce_creatures()
        dead_creatures = [creature for creature in self.creatures if creature.energy <= 0.0]
        for creature in dead_creatures:
            self._deposit_detritus(creature)
            if creature.id in self._predation_kill_ids:
                self._record_event(
                    "predation_kill",
                    creature_id=creature.id,
                    parent_ids=(),
                    energy=creature.energy,
                    genome_hash_value=genome_hash(creature.genome),
                )
            self._record_event(
                "death",
                creature_id=creature.id,
                parent_ids=(),
                energy=creature.energy,
                genome_hash_value=genome_hash(creature.genome),
            )
        living_creatures = [creature for creature in self.creatures if creature.energy > 0.0]
        if len(living_creatures) == len(self.creatures):
            self._record_speciation_events()
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
        self._predation_kill_ids = {creature_id for creature_id in self._predation_kill_ids if creature_id in {creature.id for creature in self.creatures}}
        self.grip_latches = [
            GripLatch(
                creature_a_id=latch.creature_a_id,
                node_a_index=node_index_map[latch.node_a_index],
                creature_b_id=latch.creature_b_id,
                node_b_index=node_index_map[latch.node_b_index],
                rest_length=latch.rest_length,
            )
            for latch in self.grip_latches
            if latch.node_a_index in node_index_map and latch.node_b_index in node_index_map
        ]
        self._apply_population_floor()
        self._record_speciation_events()
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
            if not self._reproduction_allowed(creature):
                updated_creatures.append(creature)
                continue

            parent_genome = creature.genome or encode_creature_genome(
                nodes=self.nodes,
                edges=self.edges,
                creature=creature,
            )
            child_genome = mutate_genome(
                genome=parent_genome,
                rng=self._rng,
                position_sigma=self.config.evolution.position_mutation_sigma,
                radius_sigma=self.config.evolution.radius_mutation_sigma,
                weight_sigma=self.config.evolution.weight_mutation_sigma,
                bias_sigma=self.config.evolution.bias_mutation_sigma,
                tau_sigma=self.config.evolution.tau_mutation_sigma,
                motor_strength_sigma=self.config.evolution.motor_strength_mutation_sigma,
                motor_toggle_mutation_rate=self.config.evolution.motor_toggle_mutation_rate,
                node_type_mutation_rate=self.config.evolution.node_type_mutation_rate,
                structural_mutation_rate=self.config.evolution.structural_mutation_rate,
                hidden_neuron_mutation_rate=self.config.evolution.hidden_neuron_mutation_rate,
                max_hidden_neurons=self.config.evolution.max_hidden_neurons,
            )
            child_offset = Vec2(2.0 * (creature_index + 1), 2.0 * (creature_index + 1))
            child_anchor = self.nodes[creature.node_indices[0]].position + child_offset
            decoded_nodes, decoded_edges, child_brain = decode_genome(
                genome=child_genome,
                anchor_position=child_anchor,
                drag_coeff=self.config.physics.default_drag,
            )
            child_node_start = len(self.nodes) + len(new_nodes)
            child_node_indices = tuple(range(child_node_start, child_node_start + len(decoded_nodes)))
            new_nodes.extend(decoded_nodes)
            new_edges.extend(
                EdgeState(
                    a=child_node_indices[edge.a],
                    b=child_node_indices[edge.b],
                    rest_length=edge.rest_length,
                    stiffness=edge.stiffness,
                    has_motor=edge.has_motor,
                    motor_strength=edge.motor_strength,
                )
                for edge in decoded_edges
            )

            split_energy = creature.energy / 2.0
            updated_creatures.append(replace(creature, energy=split_energy, genome=parent_genome))
            child_id = self._allocate_creature_id()
            new_creatures.append(
                CreatureState(
                    node_indices=child_node_indices,
                    energy=split_energy,
                    brain=child_brain,
                    genome=child_genome,
                    id=child_id,
                    parent_id=creature.id,
                )
            )
            self._record_event(
                "reproduction",
                creature_id=creature.id,
                parent_ids=(creature.id,),
                energy=split_energy,
                genome_hash_value=genome_hash(parent_genome),
            )
            self._record_event(
                "birth",
                creature_id=child_id,
                parent_ids=(creature.id,),
                energy=split_energy,
                genome_hash_value=genome_hash(child_genome),
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

    def _shift_nutrient_sources_if_due(self) -> None:
        interval = self.config.environment.nutrient_shift_interval
        if interval <= 0 or not self._nutrient_source_cells:
            return
        if (self.tick + 1) % interval != 0:
            return
        self._shift_nutrient_sources(self.config.environment.nutrient_shift_count)

    def _reseed_nutrient_sources_if_due(self) -> bool:
        interval = self.config.environment.nutrient_epoch_interval
        if interval <= 0 or self.config.environment.nutrient_source_count <= 0:
            return False
        if (self.tick + 1) % interval != 0:
            return False
        self._nutrient_source_cells = self._initialize_nutrient_sources()
        return True

    def _apply_dominance_perturbation_if_due(self) -> None:
        interval = self.config.environment.dominance_perturbation_interval
        if interval <= 0 or not self._runaway_dominance_detected:
            return
        if (self.tick + 1) % interval != 0:
            return
        if self._shift_nutrient_sources(self.config.environment.dominance_perturbation_shift_count):
            self._record_event(
                "environment_perturbation",
                creature_id=-1,
                parent_ids=(),
                energy=0.0,
                genome_hash_value="",
            )

    def _random_unoccupied_nutrient_cell(self, occupied: set[tuple[int, int]]) -> tuple[int, int]:
        while True:
            candidate = (
                self._rng.randrange(self.nutrient_grid.cols),
                self._rng.randrange(self.nutrient_grid.rows),
            )
            if candidate not in occupied:
                return candidate

    def _shift_nutrient_sources(self, shift_count: int) -> bool:
        if shift_count <= 0 or not self._nutrient_source_cells:
            return False

        source_cells = set(self._nutrient_source_cells)
        shift_count = min(shift_count, len(source_cells))
        if shift_count <= 0:
            return False

        cells = list(source_cells)
        self._rng.shuffle(cells)
        for old_cell in cells[:shift_count]:
            source_cells.remove(old_cell)
            source_cells.add(self._random_unoccupied_nutrient_cell(source_cells))
        self._nutrient_source_cells = sorted(source_cells)
        return True

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

    def _ensure_creature_genomes(self, creatures: list[CreatureState]) -> list[CreatureState]:
        ensured: list[CreatureState] = []
        for creature in creatures:
            genome = creature.genome or encode_creature_genome(
                nodes=self.nodes,
                edges=self.edges,
                creature=creature,
            )
            ensured.append(replace(creature, genome=genome))
        return ensured

    def _species_labels(self) -> dict[int, str]:
        if not self.creatures:
            return {}
        labels = cluster_species(tuple(creature.genome for creature in self.creatures))
        return {
            creature.id: label
            for creature, label in zip(self.creatures, labels, strict=True)
        }

    def _sync_species_presence(self) -> None:
        species_labels = self._species_labels()
        current_species_ids = set(species_labels.values())
        self._peak_species_count = max(self._peak_species_count, len(current_species_ids))
        self._update_species_dominance_metrics(species_labels)
        for species_id in current_species_ids:
            self._species_first_seen_tick.setdefault(species_id, self.tick)
            self._species_last_seen_tick[species_id] = self.tick
        self._known_species_ids.update(current_species_ids)

    def _record_speciation_events(self) -> None:
        species_labels = self._species_labels()
        current_species_ids = set(species_labels.values())
        new_species_ids = sorted(current_species_ids - self._known_species_ids)
        self._peak_species_count = max(self._peak_species_count, len(current_species_ids))
        self._update_species_dominance_metrics(species_labels)
        for species_id in new_species_ids:
            representative = next(
                creature
                for creature in self.creatures
                if species_labels.get(creature.id) == species_id
            )
            self._record_event(
                "speciation",
                creature_id=representative.id,
                parent_ids=(),
                energy=representative.energy,
                genome_hash_value=genome_hash(representative.genome),
            )
            self._species_first_seen_tick.setdefault(species_id, self.tick)
            self._species_last_seen_tick[species_id] = self.tick

        extinct_species_ids = sorted(
            species_id
            for species_id in (self._known_species_ids - current_species_ids)
            if species_id not in self._extinct_species_ids
        )
        for species_id in extinct_species_ids:
            self._record_event(
                "species_extinction",
                creature_id=-1,
                parent_ids=(),
                energy=0.0,
                genome_hash_value=species_id,
            )
            self._extinct_species_ids.add(species_id)
            self._species_last_seen_tick.setdefault(species_id, self.tick)

        for species_id in current_species_ids:
            self._species_last_seen_tick[species_id] = self.tick
        self._known_species_ids.update(current_species_ids)

    def _longest_species_lifespan(self) -> int:
        if not self._species_first_seen_tick:
            return 0
        return max(
            self._species_last_seen_tick.get(species_id, self.tick) - first_seen
            for species_id, first_seen in self._species_first_seen_tick.items()
        )

    def _mean_extinct_species_lifespan(self) -> float:
        if not self._extinct_species_ids:
            return 0.0
        lifespans = [
            self._species_last_seen_tick.get(species_id, self.tick) - self._species_first_seen_tick.get(species_id, 0)
            for species_id in self._extinct_species_ids
        ]
        return sum(lifespans) / len(lifespans)

    def _update_species_dominance_metrics(self, species_labels: dict[int, str]) -> None:
        if not species_labels:
            self._dominant_species_id = None
            self._dominant_species_streak = 0
            return

        species_counts = Counter(species_labels.values())
        total = sum(species_counts.values())
        dominant_species_id, dominant_count = max(species_counts.items(), key=lambda item: item[1])
        dominant_fraction = dominant_count / total if total > 0 else 0.0
        self._peak_species_fraction = max(self._peak_species_fraction, dominant_fraction)

        if dominant_fraction > self._runaway_dominance_fraction_threshold:
            if dominant_species_id == self._dominant_species_id:
                self._dominant_species_streak += 1
            else:
                self._dominant_species_id = dominant_species_id
                self._dominant_species_streak = 1
        else:
            self._dominant_species_id = None
            self._dominant_species_streak = 0

        if self._dominant_species_streak > self._runaway_dominance_tick_threshold:
            self._runaway_dominance_detected = True

    def _record_population_observation(self) -> None:
        population = len(self.creatures) if self.creatures else len(self.nodes)
        self._peak_population = max(self._peak_population, population)
        self._population_observation_count += 1
        delta = population - self._population_mean
        self._population_mean += delta / self._population_observation_count
        delta2 = population - self._population_mean
        self._population_m2 += delta * delta2

    def population_variance(self) -> float:
        if self._population_observation_count <= 1:
            return 0.0
        return self._population_m2 / self._population_observation_count

    def population_capacity_fraction(self) -> float:
        capacity = max(self.config.creatures.max_population, 1)
        return (len(self.creatures) if self.creatures else len(self.nodes)) / capacity

    def peak_population_capacity_fraction(self) -> float:
        capacity = max(self.config.creatures.max_population, 1)
        return self._peak_population / capacity

    def crowding_multiplier(self) -> float:
        capacity = max(self.config.creatures.max_population, 1)
        population = len(self.creatures) if self.creatures else len(self.nodes)
        return max(1.0, population / capacity)

    def peak_crowding_multiplier(self) -> float:
        capacity = max(self.config.creatures.max_population, 1)
        return max(1.0, self._peak_population / capacity)

    def _update_recent_speeds(self, creatures: list[CreatureState]) -> list[CreatureState]:
        updated: list[CreatureState] = []
        for creature in creatures:
            if not creature.node_indices:
                updated.append(creature)
                continue
            current_mean_speed = sum(
                self.nodes[node_index].velocity.magnitude() for node_index in creature.node_indices
            ) / len(creature.node_indices)
            updated.append(
                replace(
                    creature,
                    mean_speed_recent=(0.9 * creature.mean_speed_recent) + (0.1 * current_mean_speed),
                )
            )
        return updated

    def _reproduction_allowed(self, creature: CreatureState) -> bool:
        _, _, _, reproduce_output, _ = self._control_outputs_for_creature(creature)
        if reproduce_output is None:
            return True
        return reproduce_output >= 0.5

    def _apply_predation(self, creatures: list[CreatureState]) -> list[CreatureState]:
        if self.config.energy.predation_rate <= 0.0:
            return creatures

        energies = [creature.energy for creature in creatures]
        self._predation_kill_ids = set()
        for predator_index, predator in enumerate(creatures):
            if not self._gripper_node_indices_for_creature(predator):
                continue
            _, _, bite_outputs, _, _ = self._control_outputs_for_creature(predator)
            mouth_nodes = self._mouth_nodes_for_creature(predator)
            for mouth_node, bite_output in zip(mouth_nodes, bite_outputs, strict=False):
                if bite_output <= 0.0:
                    continue
                for victim_index, victim in enumerate(creatures):
                    if predator_index == victim_index or energies[victim_index] <= 0.0:
                        continue
                    latched_victim = self._creatures_share_grip_latch(predator, victim)
                    if not latched_victim and not any(
                        self._nodes_overlap(mouth_node, self.nodes[node_index]) for node_index in victim.node_indices
                    ):
                        continue
                    damage = min(
                        energies[victim_index],
                        bite_output * self.config.energy.predation_rate,
                    )
                    energies[victim_index] -= damage
                    energies[predator_index] += damage * self.config.energy.predation_transfer_efficiency
                    self._emit_alarm_chemicals(victim)
                    if energies[victim_index] <= 0.0:
                        self._predation_kill_ids.add(victim.id)
                    break

        return [replace(creature, energy=energy) for creature, energy in zip(creatures, energies, strict=True)]

    def _emit_chemicals(self, creatures: list[CreatureState]) -> None:
        for creature in creatures:
            _, _, _, _, chemical_outputs = self._control_outputs_for_creature(creature)
            if not chemical_outputs:
                continue
            centroid = self._creature_centroid(creature)
            if centroid is None:
                continue
            if len(chemical_outputs) >= 1 and chemical_outputs[0] > 0.0:
                self.chemical_a_grid.add_value_at_position(centroid, chemical_outputs[0])
            if len(chemical_outputs) >= 2 and chemical_outputs[1] > 0.0:
                self.chemical_b_grid.add_value_at_position(centroid, chemical_outputs[1])

    def _emit_alarm_chemicals(self, creature: CreatureState) -> None:
        for node_index in creature.node_indices:
            self.chemical_a_grid.add_value_at_position(self.nodes[node_index].position, 0.5)

    def _motor_edge_indices_for_creature(self, creature: CreatureState) -> list[int]:
        node_index_set = set(creature.node_indices)
        return [
            edge_index
            for edge_index, edge in enumerate(self.edges)
            if edge.has_motor and edge.a in node_index_set and edge.b in node_index_set
        ]

    def _mouth_nodes_for_creature(self, creature: CreatureState) -> list[NodeState]:
        return [
            self.nodes[node_index]
            for node_index in creature.node_indices
            if self.nodes[node_index].node_type == NodeType.MOUTH
        ]

    def _gripper_node_indices_for_creature(self, creature: CreatureState) -> list[int]:
        return [
            node_index
            for node_index in creature.node_indices
            if self.nodes[node_index].node_type == NodeType.GRIPPER
        ]

    def _gripper_nodes_for_creature(self, creature: CreatureState) -> list[NodeState]:
        return [self.nodes[node_index] for node_index in self._gripper_node_indices_for_creature(creature)]

    def _control_outputs_for_creature(
        self,
        creature: CreatureState,
    ) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], float | None, tuple[float, ...]]:
        outputs = creature.last_brain_outputs
        if not outputs:
            return (), (), (), None, ()

        motor_edge_count = len(self._motor_edge_indices_for_creature(creature))
        gripper_count = len(self._gripper_node_indices_for_creature(creature))
        mouth_count = len(self._mouth_nodes_for_creature(creature))
        motor_outputs = outputs[:motor_edge_count]
        grip_start = motor_edge_count
        grip_end = grip_start + gripper_count
        grip_outputs = outputs[grip_start:grip_end]
        bite_start = grip_end
        bite_end = bite_start + mouth_count
        bite_outputs = outputs[bite_start:bite_end]
        remaining_outputs = outputs[bite_end:]
        reproduce_output = remaining_outputs[0] if remaining_outputs else None
        chemical_outputs = remaining_outputs[1:3] if len(remaining_outputs) >= 3 else ()
        return motor_outputs, grip_outputs, bite_outputs, reproduce_output, chemical_outputs

    def _creature_centroid(self, creature: CreatureState) -> Vec2 | None:
        if not creature.node_indices:
            return None
        positions = [self.nodes[node_index].position for node_index in creature.node_indices]
        return Vec2(
            sum(position.x for position in positions) / len(positions),
            sum(position.y for position in positions) / len(positions),
        )

    def _gripper_contact_signal(self, creature: CreatureState) -> float:
        gripper_nodes = self._gripper_nodes_for_creature(creature)
        if not gripper_nodes:
            return 0.0
        contacts = 0
        for gripper_node in gripper_nodes:
            if any(
                other_creature.id != creature.id
                and any(
                    self._nodes_within_grip_range(gripper_node, self.nodes[node_index])
                    for node_index in other_creature.node_indices
                )
                for other_creature in self.creatures
            ):
                contacts += 1
        return contacts / len(gripper_nodes)

    def _active_grip_count(self, creature: CreatureState) -> int:
        return sum(
            1
            for latch in self.grip_latches
            if latch.creature_a_id == creature.id or latch.creature_b_id == creature.id
        )

    def _grip_active_signal(self, creature: CreatureState) -> float:
        gripper_count = len(self._gripper_node_indices_for_creature(creature))
        if gripper_count == 0:
            return 0.0
        return min(1.0, self._active_grip_count(creature) / gripper_count)

    def _refresh_grip_latches(self) -> None:
        active_creatures = {creature.id: creature for creature in self.creatures}
        active_pairs: set[tuple[int, int]] = set()
        refreshed: list[GripLatch] = []

        for latch in self.grip_latches:
            creature_a = active_creatures.get(latch.creature_a_id)
            creature_b = active_creatures.get(latch.creature_b_id)
            if creature_a is None or creature_b is None:
                continue
            if latch.node_a_index >= len(self.nodes) or latch.node_b_index >= len(self.nodes):
                continue
            if not self._latch_endpoint_is_active(creature_a, latch.node_a_index):
                continue
            if not self._latch_endpoint_is_active(creature_b, latch.node_b_index):
                continue
            latch_force = spring_force(
                self.nodes[latch.node_a_index].position,
                self.nodes[latch.node_b_index].position,
                latch.rest_length,
                self.config.physics.grip_spring_stiffness,
            )
            if latch_force.magnitude() > self.config.physics.grip_yield_force:
                continue
            refreshed.append(latch)
            active_pairs.add(tuple(sorted((latch.node_a_index, latch.node_b_index))))

        for creature in self.creatures:
            if not creature.last_brain_outputs:
                continue
            gripper_outputs = self._control_outputs_for_creature(creature)[1]
            for local_index, node_index in enumerate(self._gripper_node_indices_for_creature(creature)):
                if local_index >= len(gripper_outputs) or gripper_outputs[local_index] < 0.5:
                    continue
                if any(latch.node_a_index == node_index or latch.node_b_index == node_index for latch in refreshed):
                    continue
                for other_creature in self.creatures:
                    if other_creature.id == creature.id:
                        continue
                    for other_node_index in other_creature.node_indices:
                        pair_key = tuple(sorted((node_index, other_node_index)))
                        if pair_key in active_pairs:
                            continue
                        if any(
                            latch.node_a_index == other_node_index or latch.node_b_index == other_node_index
                            for latch in refreshed
                        ):
                            continue
                        if not self._nodes_within_grip_range(self.nodes[node_index], self.nodes[other_node_index]):
                            continue
                        refreshed.append(
                            GripLatch(
                                creature_a_id=creature.id,
                                node_a_index=node_index,
                                creature_b_id=other_creature.id,
                                node_b_index=other_node_index,
                                rest_length=(self.nodes[other_node_index].position - self.nodes[node_index].position).magnitude(),
                            )
                        )
                        active_pairs.add(pair_key)
                        break
                    if any(latch.node_a_index == node_index or latch.node_b_index == node_index for latch in refreshed):
                        break

        self.grip_latches = refreshed

    def _latch_endpoint_is_active(self, creature: CreatureState, node_index: int) -> bool:
        if node_index not in creature.node_indices:
            return False
        if self.nodes[node_index].node_type != NodeType.GRIPPER:
            return True
        return self._grip_requested(creature, node_index)

    def _creatures_share_grip_latch(self, a: CreatureState, b: CreatureState) -> bool:
        return any(
            (latch.creature_a_id == a.id and latch.creature_b_id == b.id)
            or (latch.creature_a_id == b.id and latch.creature_b_id == a.id)
            for latch in self.grip_latches
        )

    def _grip_requested(self, creature: CreatureState, node_index: int) -> bool:
        gripper_indices = self._gripper_node_indices_for_creature(creature)
        if node_index not in gripper_indices:
            return False
        local_index = gripper_indices.index(node_index)
        grip_outputs = self._control_outputs_for_creature(creature)[1]
        return local_index < len(grip_outputs) and grip_outputs[local_index] >= 0.5

    def _nodes_overlap(self, a: NodeState, b: NodeState) -> bool:
        return (a.position - b.position).magnitude() < (a.radius + b.radius)

    def _nodes_within_grip_range(self, gripper: NodeState, target: NodeState) -> bool:
        capture_distance = (gripper.radius + target.radius) * 1.5
        return (gripper.position - target.position).magnitude() < capture_distance

    def _trophic_role(self, creature: CreatureState) -> str:
        has_mouth = any(self.nodes[node_index].node_type == NodeType.MOUTH for node_index in creature.node_indices)
        has_gripper = any(self.nodes[node_index].node_type == NodeType.GRIPPER for node_index in creature.node_indices)
        has_photoreceptor = any(
            self.nodes[node_index].node_type == NodeType.PHOTORECEPTOR for node_index in creature.node_indices
        )
        motor_count = len(self._motor_edge_indices_for_creature(creature))
        mouth_count = len(self._mouth_nodes_for_creature(creature))
        output_size = creature.brain.output_size if creature.brain is not None else 0
        has_bite_channel = output_size > motor_count and mouth_count > 0

        if has_bite_channel and has_gripper:
            return "predator"
        if has_mouth:
            return "herbivore"
        if has_photoreceptor and not has_mouth:
            return "autotroph"
        return "autotroph"

    def _record_event(
        self,
        event_type: str,
        creature_id: int,
        parent_ids: tuple[int, ...],
        energy: float,
        genome_hash_value: str,
    ) -> None:
        self.events.append(
            EventRecord(
                tick=self.tick,
                event_type=event_type,
                creature_id=creature_id,
                parent_ids=parent_ids,
                energy=energy,
                genome_hash=genome_hash_value,
            )
        )

    def _serialize_creature(self, creature: CreatureState) -> dict[str, object]:
        creature_node_indices = set(creature.node_indices)
        creature_nodes = [
            {
                "position": [node.position.x, node.position.y],
                "velocity": [node.velocity.x, node.velocity.y],
                "accumulated_force": [node.accumulated_force.x, node.accumulated_force.y],
                "drag_coeff": node.drag_coeff,
                "radius": node.radius,
                "node_type": node.node_type.value,
            }
            for node in (self.nodes[index] for index in creature.node_indices)
        ]
        creature_edges = [
            {
                "a": creature.node_indices.index(edge.a),
                "b": creature.node_indices.index(edge.b),
                "rest_length": edge.rest_length,
                "stiffness": edge.stiffness,
                "has_motor": edge.has_motor,
                "motor_strength": edge.motor_strength,
            }
            for edge in self.edges
            if edge.a in creature_node_indices and edge.b in creature_node_indices
        ]
        return {
            "energy": creature.energy,
            "id": creature.id,
            "parent_id": creature.parent_id,
            "age_ticks": creature.age_ticks,
            "mean_speed_recent": creature.mean_speed_recent,
            "genome": genome_to_dict(creature.genome),
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
            "nodes": creature_nodes,
            "edges": creature_edges,
        }

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
