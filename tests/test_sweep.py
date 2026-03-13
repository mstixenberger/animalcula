import json
import subprocess
import sys
from pathlib import Path

from animalcula.analysis.sweep import aggregate_sweep_records


def test_cli_sweep_runs_parameter_grid_and_writes_results(tmp_path: Path) -> None:
    sweep_path = tmp_path / "sweep.yaml"
    out_path = tmp_path / "results.jsonl"
    sweep_path.write_text(
        "\n".join(
            [
                "energy.reproduction_threshold:",
                "  - 0.1",
                "  - 1000.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "animalcula.cli",
            "sweep",
            "--config",
            "config/default.yaml",
            "--sweep",
            str(sweep_path),
            "--ticks",
            "1",
            "--seed",
            "11",
            "--seed-demo",
            "--workers",
            "2",
            "--out",
            str(out_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert out_path.exists()
    records = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    assert records[0]["overrides"] == {"energy.reproduction_threshold": 0.1}
    assert records[1]["overrides"] == {"energy.reproduction_threshold": 1000.0}
    assert records[0]["population"] > records[1]["population"]
    assert records[0]["births"] > records[1]["births"]
    assert "peak_population" in records[0]
    assert "population_variance" in records[0]
    assert "population_capacity_fraction" in records[0]
    assert "peak_population_capacity_fraction" in records[0]
    assert "crowding_multiplier" in records[0]
    assert "peak_crowding_multiplier" in records[0]
    assert "chemical_a_total" in records[0]
    assert "chemical_b_total" in records[0]
    assert "drag_multiplier" in records[0]
    assert "nutrient_source_strength_multiplier" in records[0]
    assert "lineage_count" in records[0]
    assert "species_count" in records[0]
    assert "diversity_index" in records[0]
    assert "speciation_events" in records[0]
    assert "species_extinctions" in records[0]
    assert "species_turnover" in records[0]
    assert "longest_species_lifespan" in records[0]
    assert "mean_extinct_species_lifespan" in records[0]
    assert "peak_species_fraction" in records[0]
    assert "runaway_dominance_detected" in records[0]
    assert "predation_kills" in records[0]
    assert "environment_perturbations" in records[0]
    assert "ended_extinct" in records[0]
    assert "had_speciation" in records[0]
    assert "had_predation" in records[0]
    assert "mean_nodes_per_creature" in records[0]
    assert "observed_species_count" in records[0]
    assert "peak_species_count" in records[0]
    assert "autotroph_count" in records[0]
    assert "herbivore_count" in records[0]
    assert "predator_count" in records[0]
    assert "trophic_balance_score" in records[0]
    assert records[0]["interestingness"] > records[1]["interestingness"]
    assert "completed=2" in result.stdout


def test_aggregate_sweep_records_groups_by_override_set() -> None:
    records = [
        {
            "overrides": {"energy.reproduction_threshold": 20.0},
            "population": 5,
            "peak_population": 6,
            "population_variance": 1.5,
            "population_capacity_fraction": 0.4,
            "peak_population_capacity_fraction": 0.5,
            "crowding_multiplier": 1.0,
            "peak_crowding_multiplier": 1.0,
            "total_energy": 10.0,
            "chemical_a_total": 0.3,
            "chemical_b_total": 0.1,
            "drag_multiplier": 1.0,
            "nutrient_source_strength_multiplier": 1.0,
            "species_count": 2,
            "diversity_index": 0.9,
            "reproductions": 4,
            "deaths": 1,
            "speciation_events": 1,
            "predation_kills": 0,
            "species_extinctions": 0,
            "species_turnover": 1,
            "environment_perturbations": 0,
            "autotroph_count": 1,
            "herbivore_count": 2,
            "predator_count": 2,
            "trophic_balance_score": 0.9,
            "ended_extinct": False,
            "had_speciation": True,
            "had_predation": False,
            "observed_species_count": 3,
            "peak_species_count": 2,
            "peak_species_fraction": 0.75,
            "longest_species_lifespan": 100,
            "mean_extinct_species_lifespan": 0.0,
            "runaway_dominance_detected": False,
        },
        {
            "overrides": {"energy.reproduction_threshold": 20.0},
            "population": 3,
            "peak_population": 5,
            "population_variance": 0.5,
            "population_capacity_fraction": 0.2,
            "peak_population_capacity_fraction": 0.35,
            "crowding_multiplier": 1.2,
            "peak_crowding_multiplier": 1.4,
            "total_energy": 6.0,
            "chemical_a_total": 0.9,
            "chemical_b_total": 0.4,
            "drag_multiplier": 1.5,
            "nutrient_source_strength_multiplier": 0.75,
            "species_count": 1,
            "diversity_index": 0.4,
            "reproductions": 2,
            "deaths": 2,
            "speciation_events": 0,
            "predation_kills": 1,
            "species_extinctions": 1,
            "species_turnover": 1,
            "environment_perturbations": 1,
            "autotroph_count": 1,
            "herbivore_count": 1,
            "predator_count": 1,
            "trophic_balance_score": 1.0,
            "ended_extinct": False,
            "had_speciation": False,
            "had_predation": True,
            "observed_species_count": 2,
            "peak_species_count": 1,
            "peak_species_fraction": 1.0,
            "longest_species_lifespan": 80,
            "mean_extinct_species_lifespan": 80.0,
            "runaway_dominance_detected": True,
        },
    ]

    summaries = aggregate_sweep_records(records)

    assert len(summaries) == 1
    assert summaries[0]["runs"] == 2
    assert summaries[0]["avg_population"] == 4.0
    assert summaries[0]["avg_population_variance"] == 1.0
    assert summaries[0]["peak_population_max"] == 6
    assert summaries[0]["avg_population_capacity_fraction"] == 0.3
    assert summaries[0]["peak_population_capacity_fraction_max"] == 0.5
    assert summaries[0]["avg_crowding_multiplier"] == 1.1
    assert summaries[0]["peak_crowding_multiplier_max"] == 1.4
    assert summaries[0]["avg_diversity_index"] == 0.65
    assert summaries[0]["avg_chemical_a_total"] == 0.6
    assert summaries[0]["avg_chemical_b_total"] == 0.25
    assert summaries[0]["avg_drag_multiplier"] == 1.25
    assert summaries[0]["avg_nutrient_source_strength_multiplier"] == 0.875
    assert summaries[0]["avg_reproductions"] == 3.0
    assert summaries[0]["avg_species_turnover"] == 1.0
    assert summaries[0]["avg_environment_perturbations"] == 0.5
    assert summaries[0]["avg_autotroph_count"] == 1.0
    assert summaries[0]["avg_herbivore_count"] == 1.5
    assert summaries[0]["avg_predator_count"] == 1.5
    assert summaries[0]["avg_trophic_balance_score"] == 0.95
    assert summaries[0]["avg_observed_species_count"] == 2.5
    assert summaries[0]["peak_species_count_max"] == 2
    assert summaries[0]["peak_species_fraction_max"] == 1.0
    assert summaries[0]["had_speciation_runs"] == 1
    assert summaries[0]["had_predation_runs"] == 1
    assert summaries[0]["runaway_dominance_runs"] == 1
    assert summaries[0]["longest_species_lifespan_max"] == 100
    assert summaries[0]["avg_mean_extinct_species_lifespan"] == 40.0
