"""Seed-bank evaluation helpers for headless promotion loops."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from typing import Any

from animalcula.analysis.metrics import interestingness_score, shannon_diversity, trophic_balance_score
from animalcula.config import Config
from animalcula.sim.genome import genome_from_dict, genome_hash
from animalcula.sim.world import World


def evaluate_seed_bank(
    *,
    config_path: str,
    genomes_path: str,
    ticks: int,
    seeds: list[int],
    turbo: bool,
    workers: int = 1,
    out_path: str | None = None,
    save_top_path: str | None = None,
    top: int = 5,
) -> dict[str, Any]:
    base_config = Config.from_yaml(config_path)
    candidates = _load_seed_bank(genomes_path)
    indexed_candidates = [
        (candidate_index, candidate)
        for candidate_index, candidate in enumerate(candidates)
        if genome_from_dict(candidate.get("genome")) is not None
    ]
    jobs = [
        (candidate_index, candidate, base_config.to_dict(), run_seed, ticks, turbo)
        for candidate_index, candidate in indexed_candidates
        for run_seed in seeds
    ]

    if workers <= 1:
        run_records = [_evaluate_seed_candidate(*job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            run_records = list(executor.map(_evaluate_seed_candidate_star, jobs))

    rankings = _aggregate_seed_runs(candidates=candidates, run_records=run_records)
    report = {
        "config": config_path,
        "genomes_path": genomes_path,
        "ticks": ticks,
        "seeds": seeds,
        "candidate_count": len(indexed_candidates),
        "rankings": rankings,
    }

    if out_path is not None:
        output = Path(out_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if save_top_path is not None:
        promoted = [
            ranking["promoted_creature"]
            for ranking in rankings[: max(0, top)]
            if ranking["promoted_creature"] is not None
        ]
        output = Path(save_top_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(promoted, indent=2), encoding="utf-8")

    return report


def _evaluate_seed_candidate_star(
    args: tuple[int, dict[str, Any], dict[str, Any], int, int, bool],
) -> dict[str, Any]:
    return _evaluate_seed_candidate(*args)


def _evaluate_seed_candidate(
    candidate_index: int,
    candidate: dict[str, Any],
    base_config_dict: dict[str, Any],
    run_seed: int,
    ticks: int,
    turbo: bool,
) -> dict[str, Any]:
    config = Config.from_dict(base_config_dict)
    world = World(config=config, seed=run_seed, turbo=turbo)
    world.seed_from_exported_payload([candidate])
    world.step(ticks)
    stats = world.stats()
    genome = genome_from_dict(candidate.get("genome"))
    promoted_creature = world.top_creature_payload(n=1, metric="energy")
    return {
        "candidate_index": candidate_index,
        "seed": run_seed,
        "source_creature_id": candidate.get("id"),
        "source_energy": float(candidate.get("energy", 1.0)),
        "genome_hash": genome_hash(genome),
        "promoted_creature": promoted_creature[0] if promoted_creature else None,
        "promoted_energy": promoted_creature[0]["energy"] if promoted_creature else 0.0,
        "population": stats.population,
        "peak_population": stats.peak_population,
        "population_variance": stats.population_variance,
        "population_capacity_fraction": stats.population_capacity_fraction,
        "peak_population_capacity_fraction": stats.peak_population_capacity_fraction,
        "crowding_multiplier": stats.crowding_multiplier,
        "peak_crowding_multiplier": stats.peak_crowding_multiplier,
        "total_energy": stats.total_energy,
        "nutrient_total": stats.nutrient_total,
        "detritus_total": stats.detritus_total,
        "chemical_a_total": stats.chemical_a_total,
        "chemical_b_total": stats.chemical_b_total,
        "drag_multiplier": stats.drag_multiplier,
        "nutrient_source_strength_multiplier": stats.nutrient_source_strength_multiplier,
        "light_intensity": stats.light_intensity,
        "light_direction_degrees": stats.light_direction_degrees,
        "births": stats.births,
        "deaths": stats.deaths,
        "reproductions": stats.reproductions,
        "speciation_events": stats.speciation_events,
        "species_extinctions": stats.species_extinctions,
        "species_turnover": stats.species_turnover,
        "predation_kills": stats.predation_kills,
        "environment_perturbations": stats.environment_perturbations,
        "ended_extinct": stats.population == 0,
        "had_speciation": stats.speciation_events > 0,
        "had_predation": stats.predation_kills > 0,
        "lineage_count": stats.lineage_count,
        "species_count": stats.species_count,
        "observed_species_count": stats.observed_species_count,
        "peak_species_count": stats.peak_species_count,
        "peak_species_fraction": stats.peak_species_fraction,
        "diversity_index": stats.diversity_index,
        "mean_speed_recent": stats.mean_speed_recent,
        "active_grip_latch_count": stats.active_grip_latch_count,
        "peak_grip_latch_count": stats.peak_grip_latch_count,
        "mean_gripper_contact_signal": stats.mean_gripper_contact_signal,
        "mean_gripper_active_signal": stats.mean_gripper_active_signal,
        "mean_nodes_per_creature": stats.mean_nodes_per_creature,
        "longest_species_lifespan": stats.longest_species_lifespan,
        "mean_extinct_species_lifespan": stats.mean_extinct_species_lifespan,
        "autotroph_count": stats.autotroph_count,
        "herbivore_count": stats.herbivore_count,
        "predator_count": stats.predator_count,
        "runaway_dominance_detected": stats.runaway_dominance_detected,
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
            species_turnover=stats.species_turnover,
            observed_species_count=stats.observed_species_count,
            peak_species_fraction=stats.peak_species_fraction,
            runaway_dominance_detected=stats.runaway_dominance_detected,
            population_capacity_fraction=stats.population_capacity_fraction,
            trophic_balance=trophic_balance_score(
                autotrophs=stats.autotroph_count,
                herbivores=stats.herbivore_count,
                predators=stats.predator_count,
            ),
        ),
    }


def _aggregate_seed_runs(
    *,
    candidates: list[dict[str, Any]],
    run_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[int, dict[str, Any]] = {}

    for record in run_records:
        candidate_index = int(record["candidate_index"])
        if candidate_index not in grouped:
            grouped[candidate_index] = {
                "candidate_index": candidate_index,
                "source_creature_id": record["source_creature_id"],
                "source_energy": record["source_energy"],
                "genome_hash": record["genome_hash"],
                "runs": 0,
                "population_sum": 0,
                "population_variance_sum": 0.0,
                "peak_population_max": 0,
                "population_capacity_fraction_sum": 0.0,
                "peak_population_capacity_fraction_max": 0.0,
                "crowding_multiplier_sum": 0.0,
                "peak_crowding_multiplier_max": 0.0,
                "energy_sum": 0.0,
                "nutrient_total_sum": 0.0,
                "detritus_total_sum": 0.0,
                "chemical_a_total_sum": 0.0,
                "chemical_b_total_sum": 0.0,
                "drag_multiplier_sum": 0.0,
                "nutrient_source_strength_multiplier_sum": 0.0,
                "light_intensity_sum": 0.0,
                "light_direction_degrees_sum": 0.0,
                "births_sum": 0,
                "deaths_sum": 0,
                "reproductions_sum": 0,
                "speciation_sum": 0,
                "extinction_sum": 0,
                "turnover_sum": 0,
                "predation_sum": 0,
                "environment_perturbation_sum": 0,
                "species_sum": 0,
                "observed_species_sum": 0,
                "peak_species_sum": 0,
                "peak_species_fraction_max": 0.0,
                "diversity_sum": 0.0,
                "mean_speed_sum": 0.0,
                "active_grip_latch_sum": 0,
                "peak_grip_latch_max": 0,
                "gripper_contact_signal_sum": 0.0,
                "gripper_active_signal_sum": 0.0,
                "lineage_sum": 0,
                "complexity_sum": 0.0,
                "longest_species_lifespan_max": 0,
                "mean_extinct_species_lifespan_sum": 0.0,
                "autotroph_sum": 0,
                "herbivore_sum": 0,
                "predator_sum": 0,
                "trophic_balance_sum": 0.0,
                "interestingness_sum": 0.0,
                "ended_extinct_runs": 0,
                "had_speciation_runs": 0,
                "had_predation_runs": 0,
                "runaway_dominance_runs": 0,
                "promoted_creature": None,
                "promoted_energy": 0.0,
                "promoted_seed": None,
            }

        bucket = grouped[candidate_index]
        bucket["runs"] += 1
        bucket["population_sum"] += record["population"]
        bucket["population_variance_sum"] += record["population_variance"]
        bucket["peak_population_max"] = max(bucket["peak_population_max"], record["peak_population"])
        bucket["population_capacity_fraction_sum"] += record["population_capacity_fraction"]
        bucket["peak_population_capacity_fraction_max"] = max(
            bucket["peak_population_capacity_fraction_max"],
            record["peak_population_capacity_fraction"],
        )
        bucket["crowding_multiplier_sum"] += record["crowding_multiplier"]
        bucket["peak_crowding_multiplier_max"] = max(
            bucket["peak_crowding_multiplier_max"],
            record["peak_crowding_multiplier"],
        )
        bucket["energy_sum"] += record["total_energy"]
        bucket["nutrient_total_sum"] += record["nutrient_total"]
        bucket["detritus_total_sum"] += record["detritus_total"]
        bucket["chemical_a_total_sum"] += record["chemical_a_total"]
        bucket["chemical_b_total_sum"] += record["chemical_b_total"]
        bucket["drag_multiplier_sum"] += record["drag_multiplier"]
        bucket["nutrient_source_strength_multiplier_sum"] += record["nutrient_source_strength_multiplier"]
        bucket["light_intensity_sum"] += record["light_intensity"]
        bucket["light_direction_degrees_sum"] += record["light_direction_degrees"]
        bucket["births_sum"] += record["births"]
        bucket["deaths_sum"] += record["deaths"]
        bucket["reproductions_sum"] += record["reproductions"]
        bucket["speciation_sum"] += record["speciation_events"]
        bucket["extinction_sum"] += record["species_extinctions"]
        bucket["turnover_sum"] += record["species_turnover"]
        bucket["predation_sum"] += record["predation_kills"]
        bucket["environment_perturbation_sum"] += record["environment_perturbations"]
        bucket["species_sum"] += record["species_count"]
        bucket["observed_species_sum"] += record["observed_species_count"]
        bucket["peak_species_sum"] += record["peak_species_count"]
        bucket["peak_species_fraction_max"] = max(bucket["peak_species_fraction_max"], record["peak_species_fraction"])
        bucket["diversity_sum"] += record["diversity_index"]
        bucket["mean_speed_sum"] += record["mean_speed_recent"]
        bucket["active_grip_latch_sum"] += record["active_grip_latch_count"]
        bucket["peak_grip_latch_max"] = max(bucket["peak_grip_latch_max"], record["peak_grip_latch_count"])
        bucket["gripper_contact_signal_sum"] += record["mean_gripper_contact_signal"]
        bucket["gripper_active_signal_sum"] += record["mean_gripper_active_signal"]
        bucket["lineage_sum"] += record["lineage_count"]
        bucket["complexity_sum"] += record["mean_nodes_per_creature"]
        bucket["longest_species_lifespan_max"] = max(
            bucket["longest_species_lifespan_max"],
            record["longest_species_lifespan"],
        )
        bucket["mean_extinct_species_lifespan_sum"] += record["mean_extinct_species_lifespan"]
        bucket["autotroph_sum"] += record["autotroph_count"]
        bucket["herbivore_sum"] += record["herbivore_count"]
        bucket["predator_sum"] += record["predator_count"]
        bucket["trophic_balance_sum"] += record["trophic_balance_score"]
        bucket["interestingness_sum"] += record["interestingness"]
        bucket["ended_extinct_runs"] += 1 if record["ended_extinct"] else 0
        bucket["had_speciation_runs"] += 1 if record["had_speciation"] else 0
        bucket["had_predation_runs"] += 1 if record["had_predation"] else 0
        bucket["runaway_dominance_runs"] += 1 if record["runaway_dominance_detected"] else 0
        if (
            record["promoted_creature"] is not None
            and (
                bucket["promoted_creature"] is None
                or record["promoted_energy"] > bucket["promoted_energy"]
                or (
                    record["promoted_energy"] == bucket["promoted_energy"]
                    and record["interestingness"] > bucket["interestingness_sum"] / bucket["runs"]
                )
            )
        ):
            bucket["promoted_creature"] = record["promoted_creature"]
            bucket["promoted_energy"] = record["promoted_energy"]
            bucket["promoted_seed"] = record["seed"]

    rankings = [
        {
            "candidate_index": bucket["candidate_index"],
            "source_creature_id": bucket["source_creature_id"],
            "source_energy": bucket["source_energy"],
            "genome_hash": bucket["genome_hash"],
            "promoted_creature": bucket["promoted_creature"],
            "promoted_energy": bucket["promoted_energy"],
            "promoted_seed": bucket["promoted_seed"],
            "runs": bucket["runs"],
            "avg_population": round(bucket["population_sum"] / bucket["runs"], 3),
            "avg_population_variance": round(bucket["population_variance_sum"] / bucket["runs"], 3),
            "peak_population_max": bucket["peak_population_max"],
            "avg_population_capacity_fraction": round(
                bucket["population_capacity_fraction_sum"] / bucket["runs"],
                3,
            ),
            "peak_population_capacity_fraction_max": round(bucket["peak_population_capacity_fraction_max"], 3),
            "avg_crowding_multiplier": round(bucket["crowding_multiplier_sum"] / bucket["runs"], 3),
            "peak_crowding_multiplier_max": round(bucket["peak_crowding_multiplier_max"], 3),
            "avg_total_energy": round(bucket["energy_sum"] / bucket["runs"], 3),
            "avg_nutrient_total": round(bucket["nutrient_total_sum"] / bucket["runs"], 3),
            "avg_detritus_total": round(bucket["detritus_total_sum"] / bucket["runs"], 3),
            "avg_chemical_a_total": round(bucket["chemical_a_total_sum"] / bucket["runs"], 3),
            "avg_chemical_b_total": round(bucket["chemical_b_total_sum"] / bucket["runs"], 3),
            "avg_drag_multiplier": round(bucket["drag_multiplier_sum"] / bucket["runs"], 3),
            "avg_nutrient_source_strength_multiplier": round(
                bucket["nutrient_source_strength_multiplier_sum"] / bucket["runs"],
                3,
            ),
            "avg_light_intensity": round(bucket["light_intensity_sum"] / bucket["runs"], 3),
            "avg_light_direction_degrees": round(bucket["light_direction_degrees_sum"] / bucket["runs"], 3),
            "avg_births": round(bucket["births_sum"] / bucket["runs"], 3),
            "avg_deaths": round(bucket["deaths_sum"] / bucket["runs"], 3),
            "avg_reproductions": round(bucket["reproductions_sum"] / bucket["runs"], 3),
            "avg_speciation_events": round(bucket["speciation_sum"] / bucket["runs"], 3),
            "avg_species_extinctions": round(bucket["extinction_sum"] / bucket["runs"], 3),
            "avg_species_turnover": round(bucket["turnover_sum"] / bucket["runs"], 3),
            "avg_predation_kills": round(bucket["predation_sum"] / bucket["runs"], 3),
            "avg_environment_perturbations": round(bucket["environment_perturbation_sum"] / bucket["runs"], 3),
            "avg_species_count": round(bucket["species_sum"] / bucket["runs"], 3),
            "avg_observed_species_count": round(bucket["observed_species_sum"] / bucket["runs"], 3),
            "avg_peak_species_count": round(bucket["peak_species_sum"] / bucket["runs"], 3),
            "peak_species_fraction_max": round(bucket["peak_species_fraction_max"], 3),
            "avg_diversity_index": round(bucket["diversity_sum"] / bucket["runs"], 3),
            "avg_mean_speed_recent": round(bucket["mean_speed_sum"] / bucket["runs"], 3),
            "avg_active_grip_latch_count": round(bucket["active_grip_latch_sum"] / bucket["runs"], 3),
            "peak_grip_latch_count_max": bucket["peak_grip_latch_max"],
            "avg_mean_gripper_contact_signal": round(bucket["gripper_contact_signal_sum"] / bucket["runs"], 3),
            "avg_mean_gripper_active_signal": round(bucket["gripper_active_signal_sum"] / bucket["runs"], 3),
            "avg_lineage_count": round(bucket["lineage_sum"] / bucket["runs"], 3),
            "avg_mean_nodes_per_creature": round(bucket["complexity_sum"] / bucket["runs"], 3),
            "longest_species_lifespan_max": bucket["longest_species_lifespan_max"],
            "avg_mean_extinct_species_lifespan": round(
                bucket["mean_extinct_species_lifespan_sum"] / bucket["runs"],
                3,
            ),
            "avg_autotroph_count": round(bucket["autotroph_sum"] / bucket["runs"], 3),
            "avg_herbivore_count": round(bucket["herbivore_sum"] / bucket["runs"], 3),
            "avg_predator_count": round(bucket["predator_sum"] / bucket["runs"], 3),
            "avg_trophic_balance_score": round(bucket["trophic_balance_sum"] / bucket["runs"], 3),
            "avg_interestingness": round(bucket["interestingness_sum"] / bucket["runs"], 3),
            "ended_extinct_runs": bucket["ended_extinct_runs"],
            "had_speciation_runs": bucket["had_speciation_runs"],
            "had_predation_runs": bucket["had_predation_runs"],
            "runaway_dominance_runs": bucket["runaway_dominance_runs"],
        }
        for bucket in grouped.values()
    ]

    rankings.sort(
        key=lambda row: (
            row["ended_extinct_runs"],
            -row["avg_interestingness"],
            -row["avg_species_turnover"],
            -row["avg_observed_species_count"],
            -row["avg_population"],
            row["candidate_index"],
        )
    )
    for rank, row in enumerate(rankings, start=1):
        row["rank"] = rank

    return rankings


def promote_seed_bank(
    *,
    config_path: str,
    genomes_path: str,
    ticks: int,
    seeds: list[int],
    turbo: bool,
    rounds: int,
    top: int,
    out_dir: str,
    workers: int = 1,
) -> dict[str, Any]:
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    current_genomes_path = genomes_path
    round_reports: list[dict[str, Any]] = []
    previous_promoted_hashes: set[str] = set()
    previous_top_rank_genome_hash: str | None = None

    for round_index in range(1, max(0, rounds) + 1):
        report_path = output_dir / f"round_{round_index}.report.json"
        promoted_path = output_dir / f"round_{round_index}.promoted.json"
        input_payload = _load_seed_bank(current_genomes_path)
        report = evaluate_seed_bank(
            config_path=config_path,
            genomes_path=current_genomes_path,
            ticks=ticks,
            seeds=seeds,
            turbo=turbo,
            workers=workers,
            out_path=str(report_path),
            save_top_path=str(promoted_path),
            top=top,
        )
        promoted_payload = _load_seed_bank(str(promoted_path))
        input_hash_counts = _genome_hash_counts(input_payload)
        promoted_hash_counts = _genome_hash_counts(promoted_payload)
        input_hashes = set(input_hash_counts.keys())
        promoted_hashes = set(promoted_hash_counts.keys())
        carried_hashes = previous_promoted_hashes & promoted_hashes
        top_rank = report["rankings"][0] if report["rankings"] else None
        top_rank_genome_hash = top_rank["genome_hash"] if top_rank is not None else None
        round_reports.append(
            {
                "round": round_index,
                "input_genomes_path": current_genomes_path,
                "report_path": str(report_path),
                "promoted_genomes_path": str(promoted_path),
                "candidate_count": report["candidate_count"],
                "promoted_count": len(promoted_payload),
                "input_genome_hash_count": len(input_hashes),
                "promoted_genome_hash_count": len(promoted_hashes),
                "input_diversity_index": round(shannon_diversity(input_hash_counts), 3),
                "promoted_diversity_index": round(shannon_diversity(promoted_hash_counts), 3),
                "diversity_drift": round(
                    shannon_diversity(promoted_hash_counts) - shannon_diversity(input_hash_counts),
                    3,
                ),
                "carryover_from_previous_round_count": len(carried_hashes),
                "carryover_from_previous_round_ratio": round(
                    (len(carried_hashes) / len(previous_promoted_hashes)) if previous_promoted_hashes else 0.0,
                    3,
                ),
                "top_rank_genome_hash_matches_previous_top_rank": (
                    top_rank_genome_hash == previous_top_rank_genome_hash
                    if previous_top_rank_genome_hash is not None and top_rank_genome_hash is not None
                    else None
                ),
                "top_rank": top_rank,
            }
        )
        current_genomes_path = str(promoted_path)
        previous_promoted_hashes = promoted_hashes
        previous_top_rank_genome_hash = top_rank_genome_hash
        if not promoted_payload:
            break

    round_to_round_diversity_drift = [
        {
            "from_round": previous_round["round"],
            "to_round": current_round["round"],
            "drift": round(
                current_round["promoted_diversity_index"] - previous_round["promoted_diversity_index"],
                3,
            ),
        }
        for previous_round, current_round in zip(round_reports, round_reports[1:], strict=False)
    ]
    manifest = {
        "config": config_path,
        "initial_genomes_path": genomes_path,
        "ticks": ticks,
        "seeds": seeds,
        "rounds_requested": rounds,
        "rounds_completed": len(round_reports),
        "top": top,
        "rounds": round_reports,
        "round_to_round_diversity_drift": round_to_round_diversity_drift,
        "stable_top_rank_streak": _stable_top_rank_streak(round_reports),
        "final_genomes_path": current_genomes_path,
        "manifest_path": str(output_dir / "promotion.json"),
    }
    Path(manifest["manifest_path"]).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _load_seed_bank(path: str) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError("seed bank file must contain a list of exported creatures")
    return [item for item in payload if isinstance(item, dict)]


def _genome_hash_counts(payload: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in payload:
        genome = genome_from_dict(item.get("genome"))
        if genome is None:
            continue
        hash_value = genome_hash(genome)
        counts[hash_value] = counts.get(hash_value, 0) + 1
    return counts


def _stable_top_rank_streak(round_reports: list[dict[str, Any]]) -> int:
    best = 0
    current = 0
    previous_hash: str | None = None
    for round_report in round_reports:
        top_rank = round_report.get("top_rank")
        current_hash = None if top_rank is None else top_rank.get("genome_hash")
        if current_hash is None:
            current = 0
            previous_hash = None
            continue
        if current_hash == previous_hash:
            current += 1
        else:
            current = 1
        best = max(best, current)
        previous_hash = current_hash
    return best
