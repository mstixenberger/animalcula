"""Shared viewer payload helpers."""

from __future__ import annotations

from dataclasses import asdict

from animalcula.sim.types import Vec2
from animalcula.sim.world import World


def sample_fields(world: World, *, cols: int = 12, rows: int = 12) -> dict[str, object]:
    nutrient: list[list[float]] = []
    light: list[list[float]] = []
    chemical_a: list[list[float]] = []
    chemical_b: list[list[float]] = []
    detritus: list[list[float]] = []
    width = max(world.config.world.width, 1.0)
    height = max(world.config.world.height, 1.0)

    for row in range(rows):
        nutrient_row: list[float] = []
        light_row: list[float] = []
        chemical_a_row: list[float] = []
        chemical_b_row: list[float] = []
        detritus_row: list[float] = []
        sample_y = ((row + 0.5) / rows) * height
        for col in range(cols):
            sample_x = ((col + 0.5) / cols) * width
            position = Vec2(sample_x, sample_y)
            nutrient_row.append(world.nutrient_grid.sample(position))
            light_row.append(world.light_grid.sample(position))
            chemical_a_row.append(world.chemical_a_grid.sample(position))
            chemical_b_row.append(world.chemical_b_grid.sample(position))
            detritus_row.append(world.detritus_grid.sample(position))
        nutrient.append(nutrient_row)
        light.append(light_row)
        chemical_a.append(chemical_a_row)
        chemical_b.append(chemical_b_row)
        detritus.append(detritus_row)

    return {
        "cols": cols,
        "rows": rows,
        "nutrient": nutrient,
        "light": light,
        "chemical_a": chemical_a,
        "chemical_b": chemical_b,
        "detritus": detritus,
    }


def snapshot_payload(
    world: World,
    *,
    field_cols: int = 12,
    field_rows: int = 12,
) -> dict[str, object]:
    payload = asdict(world.snapshot())
    payload["fields"] = sample_fields(world, cols=field_cols, rows=field_rows)
    payload["stats"] = asdict(world.stats())
    return payload
