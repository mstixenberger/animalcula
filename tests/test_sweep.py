import json
import subprocess
import sys
from pathlib import Path


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
    assert records[0]["population"] == 6
    assert records[1]["population"] == 3
    assert records[0]["births"] > records[1]["births"]
    assert "lineage_count" in records[0]
    assert "species_count" in records[0]
    assert "diversity_index" in records[0]
    assert "speciation_events" in records[0]
    assert "species_extinctions" in records[0]
    assert "longest_species_lifespan" in records[0]
    assert "predation_kills" in records[0]
    assert "ended_extinct" in records[0]
    assert "had_speciation" in records[0]
    assert "had_predation" in records[0]
    assert "mean_nodes_per_creature" in records[0]
    assert "autotroph_count" in records[0]
    assert "herbivore_count" in records[0]
    assert "predator_count" in records[0]
    assert records[0]["interestingness"] > records[1]["interestingness"]
    assert "completed=2" in result.stdout
