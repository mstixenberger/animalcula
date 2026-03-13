from pathlib import Path

import subprocess
import sys

from animalcula import Config, World
from animalcula.web import app as web_app


def test_build_web_index_html_contains_live_frontend_controls() -> None:
    payload = web_app.build_web_index_html()

    assert "Animalcula Live Frontend" in payload
    assert 'new WebSocket' in payload
    assert 'id="playPause"' in payload
    assert 'id="timelineCanvas"' in payload
    assert 'id="inspectorHistory"' in payload
    assert "double-click to follow" in payload


def test_create_web_app_exposes_http_and_websocket_routes() -> None:
    world = World(config=Config.from_yaml(Path("config/default.yaml")), seed=7)
    world.seed_demo_archetypes()

    app = web_app.create_web_app(world, target_fps=24, default_speed=6)
    route_paths = {getattr(route, "path", None) for route in app.routes}

    assert "/" in route_paths
    assert "/health" in route_paths
    assert "/ws" in route_paths


def test_run_web_frontend_invokes_uvicorn_without_opening_browser_when_disabled(
    monkeypatch,
) -> None:
    world = World(config=Config.from_yaml(Path("config/default.yaml")), seed=7)
    world.seed_demo_archetypes()
    opened: list[str] = []
    captured: dict[str, object] = {}

    monkeypatch.setattr(web_app.webbrowser, "open", lambda url: opened.append(url) or True)

    def _fake_run(app: object, *, host: str, port: int, log_level: str) -> None:
        captured["app"] = app
        captured["host"] = host
        captured["port"] = port
        captured["log_level"] = log_level

    monkeypatch.setattr(web_app.uvicorn, "run", _fake_run)

    url = web_app.run_web_frontend(
        world,
        host="127.0.0.1",
        port=9876,
        target_fps=20,
        default_speed=5,
        open_browser=False,
    )

    assert url == "http://127.0.0.1:9876/"
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 9876
    assert captured["log_level"] == "warning"
    assert opened == []


def test_cli_web_help_describes_live_browser_frontend() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "animalcula.cli",
            "web",
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "live browser frontend" in result.stdout
    assert "--target-fps" in result.stdout
    assert "--default-speed" in result.stdout
    assert "--no-open-browser" in result.stdout
