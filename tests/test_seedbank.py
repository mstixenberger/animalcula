import json
from pathlib import Path

from animalcula.analysis.seedbank import evaluate_seed_bank, promote_seed_bank
from animalcula.config import Config
from animalcula.sim.world import World


def test_evaluate_seed_bank_ranks_candidates_by_aggregate_results(tmp_path: Path) -> None:
    source_world = World(config=Config.from_yaml(Path("config/default.yaml")), seed=7)
    source_world.seed_demo_archetypes()
    seed_bank_path = tmp_path / "seed-bank.json"
    promoted_path = tmp_path / "promoted.json"
    source_world.export_top_creatures(path=seed_bank_path, n=1)

    payload = json.loads(seed_bank_path.read_text(encoding="utf-8"))
    low_energy = {**payload[0], "energy": 0.1}
    high_energy = {**payload[0], "energy": 10.0}
    seed_bank_path.write_text(json.dumps([low_energy, high_energy], indent=2), encoding="utf-8")

    report = evaluate_seed_bank(
        config_path="config/default.yaml",
        genomes_path=str(seed_bank_path),
        ticks=1,
        seeds=[11, 12],
        turbo=False,
        workers=1,
        save_top_path=str(promoted_path),
        top=1,
    )

    assert report["candidate_count"] == 2
    assert report["seeds"] == [11, 12]
    assert len(report["rankings"]) == 2
    assert report["rankings"][0]["source_energy"] == 10.0
    assert report["rankings"][0]["avg_interestingness"] >= report["rankings"][1]["avg_interestingness"]
    assert "avg_population_variance" in report["rankings"][0]
    assert "peak_population_max" in report["rankings"][0]
    assert "avg_population_capacity_fraction" in report["rankings"][0]
    assert "peak_population_capacity_fraction_max" in report["rankings"][0]
    assert "avg_crowding_multiplier" in report["rankings"][0]
    assert "peak_crowding_multiplier_max" in report["rankings"][0]
    assert "avg_chemical_a_total" in report["rankings"][0]
    assert "avg_chemical_b_total" in report["rankings"][0]
    assert "avg_drag_multiplier" in report["rankings"][0]
    assert "avg_nutrient_source_strength_multiplier" in report["rankings"][0]
    assert "avg_species_turnover" in report["rankings"][0]
    assert "avg_environment_perturbations" in report["rankings"][0]
    assert "avg_observed_species_count" in report["rankings"][0]
    assert "peak_species_fraction_max" in report["rankings"][0]
    assert "runaway_dominance_runs" in report["rankings"][0]
    assert "avg_trophic_balance_score" in report["rankings"][0]

    promoted = json.loads(promoted_path.read_text(encoding="utf-8"))
    assert len(promoted) == 1
    assert report["rankings"][0]["promoted_energy"] == promoted[0]["energy"]
    assert promoted[0]["genome"] is not None
    assert promoted[0]["energy"] > 0.0


def test_promote_seed_bank_runs_multiple_rounds_and_records_manifest(tmp_path: Path) -> None:
    source_world = World(config=Config.from_yaml(Path("config/default.yaml")), seed=7)
    source_world.seed_demo_archetypes()
    seed_bank_path = tmp_path / "seed-bank.json"
    out_dir = tmp_path / "promotion"
    source_world.export_top_creatures(path=seed_bank_path, n=2)

    manifest = promote_seed_bank(
        config_path="config/default.yaml",
        genomes_path=str(seed_bank_path),
        ticks=1,
        seeds=[11],
        turbo=False,
        workers=1,
        rounds=2,
        top=1,
        out_dir=str(out_dir),
    )

    assert manifest["rounds_requested"] == 2
    assert manifest["rounds_completed"] == 2
    assert len(manifest["rounds"]) == 2
    assert manifest["rounds"][0]["candidate_count"] == 2
    assert manifest["rounds"][1]["candidate_count"] == 1
    assert manifest["rounds"][0]["input_genome_hash_count"] == 2
    assert manifest["rounds"][0]["promoted_genome_hash_count"] == 1
    assert "input_diversity_index" in manifest["rounds"][0]
    assert "promoted_diversity_index" in manifest["rounds"][0]
    assert "diversity_drift" in manifest["rounds"][0]
    assert manifest["rounds"][1]["carryover_from_previous_round_count"] == 1
    assert manifest["rounds"][1]["carryover_from_previous_round_ratio"] == 1.0
    assert manifest["rounds"][1]["top_rank_genome_hash_matches_previous_top_rank"] is True
    assert "round_to_round_diversity_drift" in manifest
    assert "stable_top_rank_streak" in manifest
    assert manifest["stable_top_rank_streak"] >= 1
    assert Path(manifest["manifest_path"]).exists()
    assert Path(manifest["final_genomes_path"]).exists()
    final_payload = json.loads(Path(manifest["final_genomes_path"]).read_text(encoding="utf-8"))
    assert len(final_payload) == 1
    assert final_payload[0]["genome"] is not None
