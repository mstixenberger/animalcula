import subprocess
import sys
from pathlib import Path

import pytest

from animalcula import Config, World
from animalcula.sim.types import CreatureState, EdgeState, NodeState, NodeType, Vec2


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
        "energy",
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


def test_world_step_applies_edge_springs_before_integration() -> None:
    config = Config.from_yaml(Path("config/default.yaml"))
    nodes = [
        NodeState(
            position=Vec2(0.0, 0.0),
            velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(),
            drag_coeff=1.0,
            radius=1.0,
        ),
        NodeState(
            position=Vec2(3.0, 0.0),
            velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(),
            drag_coeff=1.0,
            radius=1.0,
        ),
    ]
    edges = [EdgeState(a=0, b=1, rest_length=1.0, stiffness=2.0)]
    world = World(config=config, nodes=nodes, edges=edges)

    world.step()

    assert world.nodes[0].position.x > 0.0
    assert world.nodes[1].position.x < 3.0


def test_world_step_updates_creature_energy_from_light() -> None:
    config = Config.from_yaml(Path("config/default.yaml"))
    node = NodeState(
        position=Vec2(995.0, 500.0),
        velocity=Vec2.zero(),
        accumulated_force=Vec2.zero(),
        drag_coeff=1.0,
        radius=1.0,
        node_type=NodeType.PHOTORECEPTOR,
    )
    creature = CreatureState(node_indices=(0,), energy=1.0)
    world = World(config=config, nodes=[node], creatures=[creature])

    world.step()

    assert world.creatures[0].energy > 1.0


def test_world_step_updates_creature_energy_from_nutrients() -> None:
    config = Config.from_yaml(Path("config/default.yaml"))
    node = NodeState(
        position=Vec2(100.0, 100.0),
        velocity=Vec2.zero(),
        accumulated_force=Vec2.zero(),
        drag_coeff=1.0,
        radius=1.0,
        node_type=NodeType.MOUTH,
    )
    creature = CreatureState(node_indices=(0,), energy=1.0)
    world = World(config=config, nodes=[node], creatures=[creature])
    world.nutrient_grid.set_value(col=20, row=20, value=2.0)

    world.step()

    assert world.creatures[0].energy > 1.0


def test_world_removes_creature_when_energy_is_depleted() -> None:
    config = Config.from_yaml(Path("config/default.yaml"))
    node = NodeState(
        position=Vec2(10.0, 10.0),
        velocity=Vec2.zero(),
        accumulated_force=Vec2.zero(),
        drag_coeff=1.0,
        radius=1.0,
    )
    creature = CreatureState(node_indices=(0,), energy=0.0005)
    world = World(config=config, nodes=[node], creatures=[creature])

    world.step()

    assert world.creatures == []
    assert world.nodes == []


def test_world_can_seed_demo_archetypes() -> None:
    config = Config.from_yaml(Path("config/default.yaml"))
    world = World(config=config, seed=7)

    world.seed_demo_archetypes()

    assert len(world.creatures) == 3
    assert len(world.nodes) >= 5
    assert len(world.edges) >= 2


def test_demo_archetype_seeding_is_deterministic_for_seed() -> None:
    config = Config.from_yaml(Path("config/default.yaml"))
    world_a = World(config=config, seed=7)
    world_b = World(config=config, seed=7)

    world_a.seed_demo_archetypes()
    world_b.seed_demo_archetypes()

    assert world_a.nodes == world_b.nodes
    assert world_a.edges == world_b.edges
    assert world_a.creatures == world_b.creatures


def test_seeded_demo_world_can_step_without_immediate_extinction() -> None:
    config = Config.from_yaml(Path("config/default.yaml"))
    world = World(config=config, seed=7)
    world.seed_demo_archetypes()

    world.step()

    assert len(world.creatures) > 0


def test_world_reproduces_energy_rich_creatures() -> None:
    config = Config.from_yaml(Path("config/default.yaml"))
    nodes = [
        NodeState(
            position=Vec2(50.0, 50.0),
            velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(),
            drag_coeff=1.0,
            radius=1.0,
        ),
        NodeState(
            position=Vec2(56.0, 50.0),
            velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(),
            drag_coeff=1.0,
            radius=1.0,
            node_type=NodeType.PHOTORECEPTOR,
        ),
    ]
    edges = [EdgeState(a=0, b=1, rest_length=6.0, stiffness=1.0)]
    creatures = [CreatureState(node_indices=(0, 1), energy=200.0)]
    world = World(config=config, nodes=nodes, edges=edges, creatures=creatures)

    world.step()

    assert len(world.creatures) == 2
    assert len(world.nodes) == 4
    assert len(world.edges) == 2
    assert world.creatures[0].energy == world.creatures[1].energy
    assert world.nodes[2].radius != world.nodes[0].radius


def test_world_stats_report_population_nodes_and_total_energy() -> None:
    config = Config.from_yaml(Path("config/default.yaml"))
    world = World(config=config, seed=7)
    world.seed_demo_archetypes()

    stats = world.stats()

    assert stats.population == 3
    assert stats.node_count == len(world.nodes)
    assert stats.edge_count == len(world.edges)
    assert stats.total_energy > 0.0


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

    assert result.stdout.strip() == "tick=3 seed=11 population=0 nodes=0 total_energy=0.000"


def test_cli_run_command_can_seed_demo_world() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "animalcula.cli",
            "run",
            "--config",
            "config/default.yaml",
            "--ticks",
            "1",
            "--seed",
            "11",
            "--seed-demo",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "population=3" in result.stdout
