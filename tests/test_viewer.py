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
    opened: list[str] = []
    progress_updates: list[tuple[int, int]] = []

    def _raise_missing_tk() -> object:
        raise ModuleNotFoundError("_tkinter")

    monkeypatch.setattr(debug_viewer, "_load_tk", _raise_missing_tk)
    monkeypatch.setattr(debug_viewer.webbrowser, "open", lambda url: opened.append(url) or True)

    html_path = debug_viewer.launch_viewer(
        world,
        backend="auto",
        html_out_path=tmp_path / "viewer.html",
        max_frames=3,
        html_progress_callback=lambda completed, total: progress_updates.append((completed, total)),
    )

    assert html_path is not None
    assert html_path.exists()
    assert opened == [html_path.resolve().as_uri()]
    assert progress_updates == [(1, 3), (2, 3), (3, 3)]
    payload = html_path.read_text(encoding="utf-8")
    assert "Animalcula Debug Viewer" in payload
    assert "Generated from <code>animalcula view</code> HTML fallback" in payload
    assert 'id="speed"' in payload
    assert 'max="32"' in payload
    assert 'id="followToggle"' in payload
    assert 'id="ambientToggle"' in payload
    assert 'id="zoom"' in payload
    assert 'id="fieldMode"' in payload
    assert 'id="speciesStat"' in payload
    assert 'id="diversityStat"' in payload
    assert 'id="recentBirthsStat"' in payload
    assert 'id="historyCanvas"' in payload
    assert 'id="inspector"' in payload
    assert 'id="selectedSpecies"' in payload
    assert 'id="selectedSpeed"' in payload
    assert 'id="selectedEnergyDelta"' in payload
    assert 'id="predatorStat"' in payload
    assert 'function drawNodeGlyph' in payload
    assert 'function drawCreatureSilhouettes' in payload
    assert 'function drawCreatureBands' in payload
    assert 'function drawSelectedLabel' in payload
    assert 'function drawHistory' in payload
    assert 'hexToRgba' in payload
    assert "function creatureFocusScore" in payload
    assert "function rankedCreatureIds" in payload
    assert "function recentCounterDelta" in payload
    assert "function selectedTrend" in payload
    assert 'node.node_type === "gripper"' in payload
    assert '"color_rgb": [' in payload
    assert '"fields": {' in payload
    assert '"stats": {' in payload
    assert '"chemical_a": [[' in payload
    assert '"chemical_b": [[' in payload
    assert '"detritus": [[' in payload
    assert '"species_id": "species-' in payload
    assert '"genome_hash": "' in payload
    assert '"silhouette_scale": ' in payload
    assert '"glyph_scale": ' in payload
    assert '"band_count": ' in payload
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
            "--no-open-browser",
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
    assert "\"tick\": 120" in payload
    assert '"light": [[' in payload
    assert "chemical-a" in payload
    assert 'canvas.addEventListener("click"' in payload
    assert 'followToggle.addEventListener("input"' in payload
    assert 'zoom.addEventListener("input"' in payload
    assert 'event.key === "4"' in payload
    assert "preferredCreatureId(snapshot)" in payload
    assert "drawHistory(frame);" in payload
    assert "drawCreatureSilhouettes(snapshot, creatureColors, creatureVisuals)" in payload
    assert "drawCreatureBands(snapshot, nodes, visual" in payload
    assert "drawSelectedLabel(selected, sx, sy)" in payload
    assert 'node.node_type === "photoreceptor"' in payload


def test_launch_viewer_can_skip_auto_open_for_html_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = World(config=Config.from_yaml(Path("config/default.yaml")), seed=7)
    world.seed_demo_archetypes()
    opened: list[str] = []

    monkeypatch.setattr(debug_viewer.webbrowser, "open", lambda url: opened.append(url) or True)

    html_path = debug_viewer.launch_viewer(
        world,
        backend="html",
        html_out_path=tmp_path / "viewer.html",
        max_frames=2,
        open_html_in_browser=False,
    )

    assert html_path is not None
    assert html_path.exists()
    assert opened == []


def test_launch_viewer_uses_tk_safe_hex_colors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = World(config=Config.from_yaml(Path("config/default.yaml")), seed=7)
    world.seed_demo_archetypes()

    class _Canvas:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            return None

        def pack(self, **_kwargs: object) -> None:
            return None

        def bind(self, *_args: object, **_kwargs: object) -> None:
            return None

        def delete(self, *_args: object, **_kwargs: object) -> None:
            return None

        def create_rectangle(self, *_args: object, **kwargs: object) -> int:
            _assert_tk_colors(kwargs)
            return 1

        def create_oval(self, *_args: object, **kwargs: object) -> int:
            _assert_tk_colors(kwargs)
            return 1

        def create_line(self, *_args: object, **kwargs: object) -> int:
            _assert_tk_colors(kwargs)
            return 1

        def create_polygon(self, *_args: object, **kwargs: object) -> int:
            _assert_tk_colors(kwargs)
            return 1

        def create_text(self, *_args: object, **kwargs: object) -> int:
            _assert_tk_colors(kwargs)
            return 1

        def bbox(self, _item: object) -> tuple[int, int, int, int]:
            return (0, 0, 40, 12)

        def tag_raise(self, *_args: object, **_kwargs: object) -> None:
            return None

    class _Root:
        def title(self, *_args: object, **_kwargs: object) -> None:
            return None

        def configure(self, *_args: object, **_kwargs: object) -> None:
            return None

        def bind(self, *_args: object, **_kwargs: object) -> None:
            return None

        def after(self, *_args: object, **_kwargs: object) -> None:
            return None

        def mainloop(self) -> None:
            return None

    class _StringVar:
        def __init__(self, value: str = "") -> None:
            self.value = value

        def set(self, value: str) -> None:
            self.value = value

    class _Label:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            return None

        def pack(self, **_kwargs: object) -> None:
            return None

    class _Frame:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            return None

        def pack(self, **_kwargs: object) -> None:
            return None

    class _Button:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            return None

        def pack(self, **_kwargs: object) -> None:
            return None

    class _FakeTkModule:
        Canvas = _Canvas
        Button = _Button
        Frame = _Frame
        Label = _Label
        StringVar = _StringVar

        @staticmethod
        def Tk() -> _Root:
            return _Root()

    def _assert_tk_colors(kwargs: dict[str, object]) -> None:
        for key in ("fill", "outline"):
            value = kwargs.get(key)
            if isinstance(value, str):
                assert not value.startswith("rgb(")

    monkeypatch.setattr(debug_viewer, "_load_tk", lambda: _FakeTkModule)

    html_path = debug_viewer.launch_viewer(
        world,
        backend="tk",
        steps_per_frame=1,
        frame_delay_ms=1,
        canvas_width=400,
        canvas_height=400,
    )

    assert html_path is None
