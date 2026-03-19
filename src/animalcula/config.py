"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True, frozen=True)
class WorldConfig:
    width: float
    height: float
    grid_resolution: float
    boundary: str


@dataclass(slots=True, frozen=True)
class PhysicsConfig:
    dt: float
    default_drag: float
    contact_repulsion: float
    grip_spring_stiffness: float
    grip_yield_force: float
    wall_repulsion_strength: float
    wall_margin: float


@dataclass(slots=True, frozen=True)
class ObstacleConfig:
    x: float
    y: float
    radius: float


@dataclass(slots=True, frozen=True)
class EnvironmentConfig:
    nutrient_diffusion_rate: float
    nutrient_source_count: int
    nutrient_source_strength: float
    nutrient_decay_rate: float
    nutrient_emission_rate: float
    nutrient_max_density: float
    nutrient_shift_interval: int
    nutrient_shift_count: int
    nutrient_epoch_interval: int
    nutrient_epoch_strength_multipliers: tuple[float, ...]
    dominance_perturbation_interval: int
    dominance_perturbation_shift_count: int
    chemical_diffusion_rate: float
    chemical_decay_rate: float
    detritus_decay_rate: float
    detritus_recycling_rate: float
    light_intensity_max: float
    light_intensity_min: float
    light_direction: tuple[float, float]
    light_season_interval: int
    light_season_steps: int
    drag_shift_interval: int
    drag_shift_multipliers: tuple[float, ...]
    obstacles: tuple[ObstacleConfig, ...]


@dataclass(slots=True, frozen=True)
class EnergyConfig:
    basal_cost_per_node: float
    feed_rate: float
    scavenging_rate: float
    photosynthesis_rate: float
    motor_cost_per_unit: float
    grip_cost: float
    predation_rate: float
    predation_transfer_efficiency: float
    reproduction_threshold: float
    mouth_reach_bonus: float
    gripper_reach_bonus: float
    max_health: float
    health_regen_rate: float
    health_regen_cost: float
    bite_health_damage: float


@dataclass(slots=True, frozen=True)
class EvolutionConfig:
    position_mutation_sigma: float
    radius_mutation_sigma: float
    weight_mutation_sigma: float
    bias_mutation_sigma: float
    tau_mutation_sigma: float
    motor_strength_mutation_sigma: float
    motor_toggle_mutation_rate: float
    node_type_mutation_rate: float
    structural_mutation_rate: float
    hidden_neuron_mutation_rate: float
    max_hidden_neurons: int
    drag_mutation_sigma: float
    chain_extension_mutation_rate: float
    max_nodes_per_creature: int
    remove_node_mutation_rate: float
    remove_edge_mutation_rate: float
    add_edge_mutation_rate: float


@dataclass(slots=True, frozen=True)
class BrainConfig:
    default_input_size: int
    motor_force_scale: float


@dataclass(slots=True, frozen=True)
class CreaturesConfig:
    min_population: int
    max_population: int


@dataclass(slots=True, frozen=True)
class SimulationConfig:
    initial_seed: int


@dataclass(slots=True, frozen=True)
class Config:
    world: WorldConfig
    physics: PhysicsConfig
    environment: EnvironmentConfig
    energy: EnergyConfig
    evolution: EvolutionConfig
    brain: BrainConfig
    creatures: CreaturesConfig
    simulation: SimulationConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        raw = _load_yaml(path)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        evolution_raw = {
            "hidden_neuron_mutation_rate": 0.0,
            "max_hidden_neurons": 24,
            "drag_mutation_sigma": 0.0,
            "chain_extension_mutation_rate": 0.0,
            "max_nodes_per_creature": 16,
            "remove_node_mutation_rate": 0.0,
            "remove_edge_mutation_rate": 0.0,
            "add_edge_mutation_rate": 0.0,
            **raw["evolution"],
        }
        environment_raw = {
            "nutrient_shift_interval": 0,
            "nutrient_shift_count": 1,
            "nutrient_epoch_interval": 0,
            "nutrient_epoch_strength_multipliers": [1.0],
            "dominance_perturbation_interval": 0,
            "dominance_perturbation_shift_count": 0,
            **raw["environment"],
        }
        environment_raw.setdefault("light_intensity_min", environment_raw["light_intensity_max"])
        environment_raw.setdefault("light_season_interval", 0)
        environment_raw.setdefault("light_season_steps", 1)
        environment_raw.setdefault("drag_shift_interval", 0)
        environment_raw.setdefault("drag_shift_multipliers", [1.0])
        environment_raw.setdefault(
            "nutrient_emission_rate",
            environment_raw.get("nutrient_source_strength", 2.0),
        )
        environment_raw.setdefault("nutrient_max_density", 10.0)
        raw_obstacles = environment_raw.pop("obstacles", [])
        obstacles = tuple(
            ObstacleConfig(x=o["x"], y=o["y"], radius=o["radius"])
            for o in raw_obstacles
        )
        return cls(
            world=WorldConfig(**raw["world"]),
            physics=PhysicsConfig(
                **{
                    "wall_repulsion_strength": 0.0,
                    "wall_margin": 0.0,
                    **raw["physics"],
                }
            ),
            environment=EnvironmentConfig(
                **{
                    **environment_raw,
                    "light_direction": tuple(environment_raw["light_direction"]),
                    "nutrient_epoch_strength_multipliers": tuple(
                        environment_raw["nutrient_epoch_strength_multipliers"]
                    ),
                    "drag_shift_multipliers": tuple(environment_raw["drag_shift_multipliers"]),
                    "obstacles": obstacles,
                }
            ),
            energy=EnergyConfig(
                **{
                    "mouth_reach_bonus": 0.0,
                    "gripper_reach_bonus": 0.0,
                    "max_health": 0.0,
                    "health_regen_rate": 0.0,
                    "health_regen_cost": 0.0,
                    "bite_health_damage": 0.0,
                    **raw["energy"],
                }
            ),
            evolution=EvolutionConfig(**evolution_raw),
            brain=BrainConfig(**raw["brain"]),
            creatures=CreaturesConfig(**raw["creatures"]),
            simulation=SimulationConfig(**raw["simulation"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_overrides(self, overrides: list[str]) -> "Config":
        raw = self.to_dict()
        for override in overrides:
            path, raw_value = override.split("=", maxsplit=1)
            value = yaml.safe_load(raw_value)
            cursor: dict[str, Any] = raw
            keys = path.split(".")
            for key in keys[:-1]:
                cursor = cursor[key]
            cursor[keys[-1]] = value
        return Config.from_dict(raw)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        msg = f"configuration root must be a mapping: {path}"
        raise TypeError(msg)

    return data
