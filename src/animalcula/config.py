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


@dataclass(slots=True, frozen=True)
class EnvironmentConfig:
    nutrient_diffusion_rate: float
    nutrient_source_count: int
    nutrient_source_strength: float
    nutrient_decay_rate: float
    light_intensity_max: float
    light_direction: tuple[float, float]


@dataclass(slots=True, frozen=True)
class EnergyConfig:
    basal_cost_per_node: float
    feed_rate: float
    photosynthesis_rate: float


@dataclass(slots=True, frozen=True)
class SimulationConfig:
    initial_seed: int


@dataclass(slots=True, frozen=True)
class Config:
    world: WorldConfig
    physics: PhysicsConfig
    environment: EnvironmentConfig
    energy: EnergyConfig
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
            simulation=SimulationConfig(**raw["simulation"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        msg = f"configuration root must be a mapping: {path}"
        raise TypeError(msg)

    return data
