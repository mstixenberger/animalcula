import json
import subprocess
import sys
from pathlib import Path


def test_tune_phase1_writes_raw_and_summary_outputs(tmp_path: Path) -> None:
    sweep_path = tmp_path / "phase1.yaml"
    out_path = tmp_path / "phase1.jsonl"
    summary_path = out_path.with_suffix(".summary.json")
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
            "scripts/tune_phase1.py",
            "--config",
            "config/default.yaml",
            "--sweep",
            str(sweep_path),
            "--ticks",
            "1",
            "--seeds",
            "11,12",
            "--workers",
            "2",
            "--out",
            str(out_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    records = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 4
    assert [record["run"] for record in records] == [1, 2, 3, 4]
    assert {record["seed"] for record in records} == {11, 12}
    assert all(record["run_count"] == 4 for record in records)
    assert summary_path.exists()
    summaries = json.loads(summary_path.read_text(encoding="utf-8"))
    assert len(summaries) == 2
    assert "avg_drag_multiplier" in summaries[0]
    assert "avg_species_turnover" in summaries[0]
    assert "avg_observed_species_count" in summaries[0]
    assert "peak_species_count_max" in summaries[0]
    assert "avg_mean_extinct_species_lifespan" in summaries[0]
    assert "saved=" in result.stdout


def test_tune_phase1_can_export_and_promote_top_seed_bank(tmp_path: Path) -> None:
    sweep_path = tmp_path / "phase1.yaml"
    out_path = tmp_path / "phase1.jsonl"
    seed_bank_path = tmp_path / "phase1.seedbank.json"
    seed_bank_manifest_path = tmp_path / "phase1.seedbank.manifest.json"
    promote_dir = tmp_path / "promotion"
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
            "scripts/tune_phase1.py",
            "--config",
            "config/default.yaml",
            "--sweep",
            str(sweep_path),
            "--ticks",
            "1",
            "--seeds",
            "11",
            "--workers",
            "1",
            "--out",
            str(out_path),
            "--save-top",
            str(seed_bank_path),
            "--top-runs",
            "1",
            "--top-creatures",
            "1",
            "--promote-out-dir",
            str(promote_dir),
            "--promote-rounds",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    seed_bank = json.loads(seed_bank_path.read_text(encoding="utf-8"))
    assert len(seed_bank) == 1
    assert seed_bank[0]["genome"] is not None
    assert seed_bank[0]["source_run"] == 1
    assert seed_bank[0]["source_seed"] == 11
    assert "source_overrides" in seed_bank[0]

    seed_bank_manifest = json.loads(seed_bank_manifest_path.read_text(encoding="utf-8"))
    assert seed_bank_manifest["selected_run_count"] == 1
    assert seed_bank_manifest["exported_creature_count"] == 1
    assert seed_bank_manifest["runs"][0]["exported_creatures"] == 1

    promotion_manifest = json.loads((promote_dir / "promotion.json").read_text(encoding="utf-8"))
    assert promotion_manifest["rounds_requested"] == 2
    assert promotion_manifest["rounds_completed"] >= 1
    assert promotion_manifest["initial_genomes_path"] == str(seed_bank_path)
    assert "promotion=" in result.stdout
