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


@dataclass(slots=True, frozen=True)
class EnvironmentConfig:
    nutrient_diffusion_rate: float
    nutrient_source_count: int
    nutrient_source_strength: float
    nutrient_decay_rate: float
    detritus_decay_rate: float
    detritus_recycling_rate: float
    light_intensity_max: float
    light_direction: tuple[float, float]


@dataclass(slots=True, frozen=True)
class EnergyConfig:
    basal_cost_per_node: float
    feed_rate: float
    scavenging_rate: float
    photosynthesis_rate: float
    motor_cost_per_unit: float
    reproduction_threshold: float


@dataclass(slots=True, frozen=True)
class EvolutionConfig:
    position_mutation_sigma: float
    radius_mutation_sigma: float
    weight_mutation_sigma: float
    bias_mutation_sigma: float
    tau_mutation_sigma: float
    motor_strength_mutation_sigma: float


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
        return cls(
            world=WorldConfig(**raw["world"]),
            physics=PhysicsConfig(**raw["physics"]),
            environment=EnvironmentConfig(
                **{
                    **raw["environment"],
                    "light_direction": tuple(raw["environment"]["light_direction"]),
                }
            ),
            energy=EnergyConfig(**raw["energy"]),
            evolution=EvolutionConfig(**raw["evolution"]),
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
