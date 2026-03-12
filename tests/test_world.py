import subprocess
import sys
from pathlib import Path

import pytest

from animalcula import Config, World
from animalcula.sim.types import NodeState, Vec2


def test_world_uses_default_seed_when_none_is_provided() -> None:
    config = Config.from_yaml(Path("config/default.yaml"))
    world = World(config=config)

    assert world.seed == 42
    assert world.tick == 0


def test_world_step_advances_tick_and_returns_snapshot() -> None:
    config = Config.from_yaml(Path("config/default.yaml"))
    world = World(config=config, seed=7)

    snapshot = world.step(5)

    assert world.tick == 5
    assert snapshot.tick == 5
    assert snapshot.population == 0
    assert snapshot.phase_trace == [
        "environment",
        "sensing",
        "brain",
        "physics",
        "lifecycle",
    ]


def test_world_rejects_negative_steps() -> None:
    config = Config.from_yaml(Path("config/default.yaml"))
    world = World(config=config)

    with pytest.raises(ValueError, match="ticks must be non-negative"):
        world.step(-1)


def test_world_random_stream_is_deterministic_for_the_same_seed() -> None:
    config = Config.from_yaml(Path("config/default.yaml"))
    world_a = World(config=config, seed=99)
    world_b = World(config=config, seed=99)

    assert world_a.random_unit() == world_b.random_unit()


def test_world_step_applies_overdamped_physics_to_nodes() -> None:
    config = Config.from_yaml(Path("config/default.yaml"))
    node = NodeState(
        position=Vec2.zero(),
        velocity=Vec2.zero(),
        accumulated_force=Vec2(1.0, 0.0),
        drag_coeff=2.0,
        radius=1.0,
    )
    world = World(config=config, nodes=[node])

    snapshot = world.step()

    assert world.nodes[0].velocity == Vec2(0.5, 0.0)
    assert world.nodes[0].position == Vec2(0.005, 0.0)
    assert snapshot.population == 1


def test_cli_run_command_advances_the_world() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "animalcula.cli",
            "run",
            "--config",
            "config/default.yaml",
            "--ticks",
            "3",
            "--seed",
            "11",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "tick=3 seed=11"
