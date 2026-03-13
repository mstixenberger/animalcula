"""Sequential parameter sweep helpers."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from pathlib import Path
from typing import Any

import yaml

from animalcula.analysis.metrics import interestingness_score, trophic_balance_score
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
    workers: int = 1,
) -> int:
    base_config = Config.from_yaml(config_path)
    sweep = _load_sweep(sweep_path)
    combinations = list(_iter_combinations(sweep))
    indexed_combinations = list(enumerate(combinations))

    if workers <= 1:
        indexed_results = [
            _run_sweep_combination(
                index=index,
                overrides=overrides,
                base_config_dict=base_config.to_dict(),
                ticks=ticks,
                seed=seed,
                seed_demo=seed_demo,
                turbo=turbo,
            )
            for index, overrides in indexed_combinations
        ]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            indexed_results = list(
                executor.map(
                    _run_sweep_combination_star,
                    [
                        (index, overrides, base_config.to_dict(), ticks, seed, seed_demo, turbo)
                        for index, overrides in indexed_combinations
                    ],
                )
            )

    results = [
        record
        for _, record in sorted(indexed_results, key=lambda item: item[0])
    ]

    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(record) + "\n" for record in results),
        encoding="utf-8",
    )
    return len(results)


def aggregate_sweep_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}

    for record in records:
        key = json.dumps(record["overrides"], sort_keys=True)
        if key not in grouped:
            grouped[key] = {
                "overrides": record["overrides"],
                "runs": 0,
                "population_sum": 0,
                "energy_sum": 0.0,
                "species_sum": 0,
                "diversity_sum": 0.0,
                "reproductions_sum": 0,
                "deaths_sum": 0,
                "speciation_sum": 0,
                "predation_sum": 0,
                "extinction_sum": 0,
                "turnover_sum": 0,
                "observed_species_sum": 0,
                "peak_species_count_max": 0,
                "autotroph_sum": 0,
                "herbivore_sum": 0,
                "predator_sum": 0,
                "trophic_balance_sum": 0.0,
                "ended_extinct_runs": 0,
                "had_speciation_runs": 0,
                "had_predation_runs": 0,
                "longest_species_lifespan_max": 0,
                "mean_extinct_species_lifespan_sum": 0.0,
            }
        bucket = grouped[key]
        bucket["runs"] += 1
        bucket["population_sum"] += record["population"]
        bucket["energy_sum"] += record["total_energy"]
        bucket["species_sum"] += record["species_count"]
        bucket["diversity_sum"] += record["diversity_index"]
        bucket["reproductions_sum"] += record["reproductions"]
        bucket["deaths_sum"] += record["deaths"]
        bucket["speciation_sum"] += record["speciation_events"]
        bucket["predation_sum"] += record["predation_kills"]
        bucket["extinction_sum"] += record["species_extinctions"]
        bucket["turnover_sum"] += record["species_turnover"]
        bucket["observed_species_sum"] += record["observed_species_count"]
        bucket["peak_species_count_max"] = max(bucket["peak_species_count_max"], record["peak_species_count"])
        bucket["autotroph_sum"] += record["autotroph_count"]
        bucket["herbivore_sum"] += record["herbivore_count"]
        bucket["predator_sum"] += record["predator_count"]
        bucket["trophic_balance_sum"] += record["trophic_balance_score"]
        bucket["ended_extinct_runs"] += 1 if record["ended_extinct"] else 0
        bucket["had_speciation_runs"] += 1 if record["had_speciation"] else 0
        bucket["had_predation_runs"] += 1 if record["had_predation"] else 0
        bucket["longest_species_lifespan_max"] = max(
            bucket["longest_species_lifespan_max"],
            record["longest_species_lifespan"],
        )
        bucket["mean_extinct_species_lifespan_sum"] += record["mean_extinct_species_lifespan"]

    summaries = [
        {
            "overrides": bucket["overrides"],
            "runs": bucket["runs"],
            "avg_population": round(bucket["population_sum"] / bucket["runs"], 3),
            "avg_total_energy": round(bucket["energy_sum"] / bucket["runs"], 3),
            "avg_species_count": round(bucket["species_sum"] / bucket["runs"], 3),
            "avg_diversity_index": round(bucket["diversity_sum"] / bucket["runs"], 3),
            "avg_reproductions": round(bucket["reproductions_sum"] / bucket["runs"], 3),
            "avg_deaths": round(bucket["deaths_sum"] / bucket["runs"], 3),
            "avg_speciation_events": round(bucket["speciation_sum"] / bucket["runs"], 3),
            "avg_predation_kills": round(bucket["predation_sum"] / bucket["runs"], 3),
            "avg_species_extinctions": round(bucket["extinction_sum"] / bucket["runs"], 3),
            "avg_species_turnover": round(bucket["turnover_sum"] / bucket["runs"], 3),
            "avg_observed_species_count": round(bucket["observed_species_sum"] / bucket["runs"], 3),
            "avg_autotroph_count": round(bucket["autotroph_sum"] / bucket["runs"], 3),
            "avg_herbivore_count": round(bucket["herbivore_sum"] / bucket["runs"], 3),
            "avg_predator_count": round(bucket["predator_sum"] / bucket["runs"], 3),
            "avg_trophic_balance_score": round(bucket["trophic_balance_sum"] / bucket["runs"], 3),
            "ended_extinct_runs": bucket["ended_extinct_runs"],
            "had_speciation_runs": bucket["had_speciation_runs"],
            "had_predation_runs": bucket["had_predation_runs"],
            "longest_species_lifespan_max": bucket["longest_species_lifespan_max"],
            "peak_species_count_max": bucket["peak_species_count_max"],
            "avg_mean_extinct_species_lifespan": round(
                bucket["mean_extinct_species_lifespan_sum"] / bucket["runs"],
                3,
            ),
        }
        for bucket in grouped.values()
    ]

    return sorted(
        summaries,
        key=lambda row: (
            row["ended_extinct_runs"],
            -row["avg_reproductions"],
            -row["avg_population"],
            -row["avg_species_count"],
            -row["avg_total_energy"],
        ),
    )


def _run_sweep_combination_star(
    args: tuple[int, dict[str, Any], dict[str, Any], int, int | None, bool, bool],
) -> tuple[int, dict[str, Any]]:
    return _run_sweep_combination(*args)


def _run_sweep_combination(
    index: int,
    overrides: dict[str, Any],
    base_config_dict: dict[str, Any],
    ticks: int,
    seed: int | None,
    seed_demo: bool,
    turbo: bool,
) -> tuple[int, dict[str, Any]]:
    config = Config.from_dict(base_config_dict).with_overrides([f"{key}={value}" for key, value in overrides.items()])
    world = World(config=config, seed=seed, turbo=turbo)
    if seed_demo:
        world.seed_demo_archetypes()
    world.step(ticks)
    stats = world.stats()
    return index, {
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
        "species_turnover": stats.species_turnover,
        "predation_kills": stats.predation_kills,
        "ended_extinct": stats.population == 0,
        "had_speciation": stats.speciation_events > 0,
        "had_predation": stats.predation_kills > 0,
        "lineage_count": stats.lineage_count,
        "species_count": stats.species_count,
        "observed_species_count": stats.observed_species_count,
        "peak_species_count": stats.peak_species_count,
        "diversity_index": stats.diversity_index,
        "mean_nodes_per_creature": stats.mean_nodes_per_creature,
        "longest_species_lifespan": stats.longest_species_lifespan,
        "mean_extinct_species_lifespan": stats.mean_extinct_species_lifespan,
        "autotroph_count": stats.autotroph_count,
        "herbivore_count": stats.herbivore_count,
        "predator_count": stats.predator_count,
        "trophic_balance_score": trophic_balance_score(
            autotrophs=stats.autotroph_count,
            herbivores=stats.herbivore_count,
            predators=stats.predator_count,
        ),
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


def _load_sweep(path: str) -> dict[str, list[Any]]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("sweep file root must be a mapping")
    return data


def _iter_combinations(sweep: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(sweep.keys())
    values = [sweep[key] for key in keys]
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*values)]
