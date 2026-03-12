"""Sequential parameter sweep helpers."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import yaml

from animalcula.analysis.metrics import interestingness_score
from animalcula.config import Config
from animalcula.sim.world import World


def run_sweep(
    *,
    config_path: str,
    sweep_path: str,
    ticks: int,
    seed: int | None,
    seed_demo: bool,
    out_path: str,
    turbo: bool,
) -> int:
    base_config = Config.from_yaml(config_path)
    sweep = _load_sweep(sweep_path)
    combinations = list(_iter_combinations(sweep))
    results: list[dict[str, Any]] = []

    for overrides in combinations:
        config = base_config.with_overrides([f"{key}={value}" for key, value in overrides.items()])
        world = World(config=config, seed=seed, turbo=turbo)
        if seed_demo:
            world.seed_demo_archetypes()
        world.step(ticks)
        stats = world.stats()
        results.append(
            {
                "overrides": overrides,
                "tick": stats.tick,
                "population": stats.population,
                "nodes": stats.node_count,
                "total_energy": stats.total_energy,
                "births": stats.births,
                "deaths": stats.deaths,
                "reproductions": stats.reproductions,
                "speciation_events": stats.speciation_events,
                "species_extinctions": stats.species_extinctions,
                "predation_kills": stats.predation_kills,
                "ended_extinct": stats.population == 0,
                "had_speciation": stats.speciation_events > 0,
                "had_predation": stats.predation_kills > 0,
                "lineage_count": stats.lineage_count,
                "species_count": stats.species_count,
                "diversity_index": stats.diversity_index,
                "mean_nodes_per_creature": stats.mean_nodes_per_creature,
                "longest_species_lifespan": stats.longest_species_lifespan,
                "autotroph_count": stats.autotroph_count,
                "herbivore_count": stats.herbivore_count,
                "predator_count": stats.predator_count,
                "interestingness": interestingness_score(
                    population=stats.population,
                    total_energy=stats.total_energy,
                    births=stats.births,
                    deaths=stats.deaths,
                    reproductions=stats.reproductions,
                    speciation_events=stats.speciation_events,
                    species_extinctions=stats.species_extinctions,
                    predation_kills=stats.predation_kills,
                ),
            }
        )

    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(record) + "\n" for record in results),
        encoding="utf-8",
    )
    return len(results)


def _load_sweep(path: str) -> dict[str, list[Any]]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("sweep file root must be a mapping")
    return data


def _iter_combinations(sweep: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(sweep.keys())
    values = [sweep[key] for key in keys]
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*values)]
