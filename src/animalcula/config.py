"""Configuration loading and validation."""

from __future__ import annotations

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
class SimulationConfig:
    initial_seed: int


@dataclass(slots=True, frozen=True)
class Config:
    world: WorldConfig
    physics: PhysicsConfig
    simulation: SimulationConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        raw = _load_yaml(path)
        return cls(
            world=WorldConfig(**raw["world"]),
            physics=PhysicsConfig(**raw["physics"]),
            simulation=SimulationConfig(**raw["simulation"]),
        )


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        msg = f"configuration root must be a mapping: {path}"
        raise TypeError(msg)

    return data
