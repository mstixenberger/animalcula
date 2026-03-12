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
    assert "saved=" in result.stdout
