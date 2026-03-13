import subprocess
import sys
from pathlib import Path

import pytest

from animalcula import Config, World
from animalcula.viz import debug_viewer


def test_launch_viewer_falls_back_to_html_when_tk_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = World(config=Config.from_yaml(Path("config/default.yaml")), seed=7)
    world.seed_demo_archetypes()

    def _raise_missing_tk() -> object:
        raise ModuleNotFoundError("_tkinter")

    monkeypatch.setattr(debug_viewer, "_load_tk", _raise_missing_tk)

    html_path = debug_viewer.launch_viewer(
        world,
        backend="auto",
        html_out_path=tmp_path / "viewer.html",
        max_frames=3,
    )

    assert html_path is not None
    assert html_path.exists()
    payload = html_path.read_text(encoding="utf-8")
    assert "Animalcula Debug Viewer" in payload
    assert "Generated from <code>animalcula view</code> HTML fallback" in payload
    assert 'id="speed"' in payload
    assert 'max="32"' in payload
    assert 'id="followToggle"' in payload
    assert 'id="ambientToggle"' in payload
    assert 'id="zoom"' in payload
    assert 'id="fieldMode"' in payload
    assert 'id="inspector"' in payload
    assert 'id="selectedSpecies"' in payload
    assert 'id="predatorStat"' in payload
    assert '"color_rgb": [' in payload
    assert '"fields": {' in payload
    assert '"chemical_a": [[' in payload
    assert '"chemical_b": [[' in payload
    assert '"detritus": [[' in payload
    assert '"species_id": "species-' in payload
    assert '"genome_hash": "' in payload
    assert "\"tick\": 0" in payload


def test_cli_view_can_write_html_viewer_without_tk(tmp_path: Path) -> None:
    html_path = tmp_path / "viewer.html"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "animalcula.cli",
            "view",
            "--config",
            "config/default.yaml",
            "--seed",
            "42",
            "--seed-demo",
            "--viewer-backend",
            "html",
            "--html-out",
            str(html_path),
            "--max-frames",
            "3",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert html_path.exists()
    assert "saved_html_viewer=" in result.stdout
    payload = html_path.read_text(encoding="utf-8")
    assert "Animalcula Debug Viewer" in payload
    assert "speed=" in payload
    assert '"light": [[' in payload
    assert "chemical-a" in payload
    assert 'canvas.addEventListener("click"' in payload
    assert 'followToggle.addEventListener("input"' in payload
    assert 'zoom.addEventListener("input"' in payload
    assert 'event.key === "4"' in payload
