import json
import subprocess
import sys
from pathlib import Path

import pytest

from animalcula import Config, World
from animalcula.sim.genome import decode_genome
from animalcula.sim.types import BrainState, CreatureState, EdgeState, NodeState, NodeType, Vec2


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


def test_world_sensing_tracks_field_gradients() -> None:
    config = Config.from_yaml(Path("config/default.yaml"))
    nodes = [
        NodeState(
            position=Vec2(500.0, 500.0),
            velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(),
            drag_coeff=1.0,
            radius=1.0,
            node_type=NodeType.PHOTORECEPTOR,
        ),
        NodeState(
            position=Vec2(100.0, 100.0),
            velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(),
            drag_coeff=1.0,
            radius=1.0,
            node_type=NodeType.MOUTH,
        ),
    ]
    creature = CreatureState(node_indices=(0, 1), energy=1.0)
    world = World(config=config, nodes=nodes, creatures=[creature])
    world.nutrient_grid.set_value(col=19, row=20, value=1.0)
    world.nutrient_grid.set_value(col=21, row=20, value=3.0)

    world.step()

    sensed = world.creatures[0].last_sensed_inputs
    assert len(sensed) == 8
    assert sensed[3] > 0.0
    assert sensed[5] > 0.0
    assert sensed[7] > 0.0


def test_world_increments_creature_age_each_tick() -> None:
    config = Config.from_yaml(Path("config/default.yaml"))
    node = NodeState(
        position=Vec2(10.0, 10.0),
        velocity=Vec2.zero(),
        accumulated_force=Vec2.zero(),
        drag_coeff=1.0,
        radius=1.0,
    )
    creature = CreatureState(node_indices=(0,), energy=1.0)
    world = World(config=config, nodes=[node], creatures=[creature])

    world.step(2)

    assert world.creatures[0].age_ticks == 2


def test_world_applies_crowding_pressure_above_population_cap() -> None:
    config = Config.from_yaml(Path("config/default.yaml")).with_overrides(["creatures.max_population=1"])
    creatures = [
        CreatureState(
            node_indices=(0,),
            energy=1.0,
            brain=None,
        ),
        CreatureState(
            node_indices=(1,),
            energy=1.0,
            brain=None,
        ),
    ]
    nodes = [
        NodeState(
            position=Vec2(10.0, 10.0),
            velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(),
            drag_coeff=1.0,
            radius=1.0,
        ),
        NodeState(
            position=Vec2(20.0, 20.0),
            velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(),
            drag_coeff=1.0,
            radius=1.0,
        ),
    ]
    world = World(config=config, nodes=nodes, creatures=creatures)

    world.step()

    assert world.creatures[0].energy < 0.999
    assert world.creatures[1].energy < 0.999


def test_world_brain_phase_updates_brain_state_and_moves_creature() -> None:
    config = Config.from_yaml(Path("config/default.yaml"))
    brain = BrainState(
        input_weights=((2.0, 0.0, 0.0),),
        recurrent_weights=((0.0,),),
        biases=(0.0,),
        time_constants=(1.0,),
        states=(0.0,),
        output_size=1,
    )
    node = NodeState(
        position=Vec2(995.0, 500.0),
        velocity=Vec2.zero(),
        accumulated_force=Vec2.zero(),
        drag_coeff=1.0,
        radius=1.0,
        node_type=NodeType.PHOTORECEPTOR,
    )
    creature = CreatureState(node_indices=(0,), energy=1.0, brain=brain)
    world = World(config=config, nodes=[node], creatures=[creature])

    world.step()

    assert world.creatures[0].brain is not None
    assert world.creatures[0].brain.states[0] > 0.0
    assert world.creatures[0].last_sensed_inputs[0] > 0.0
    assert world.nodes[0].position.x != 995.0


def test_world_brain_outputs_can_drive_motorized_edges() -> None:
    config = Config.from_yaml(Path("config/default.yaml"))
    brain = BrainState(
        input_weights=((2.0, 0.0, 0.0),),
        recurrent_weights=((0.0,),),
        biases=(0.0,),
        time_constants=(1.0,),
        states=(0.0,),
        output_size=1,
    )
    nodes = [
        NodeState(
            position=Vec2(995.0, 500.0),
            velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(),
            drag_coeff=1.0,
            radius=1.0,
            node_type=NodeType.PHOTORECEPTOR,
        ),
        NodeState(
            position=Vec2(989.0, 500.0),
            velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(),
            drag_coeff=1.0,
            radius=1.0,
        ),
    ]
    edges = [EdgeState(a=0, b=1, rest_length=6.0, stiffness=1.0, has_motor=True, motor_strength=5.0)]
    creature = CreatureState(node_indices=(0, 1), energy=1.0, brain=brain)
    world = World(config=config, nodes=nodes, edges=edges, creatures=[creature])

    world.step()

    assert world.nodes[0].position.x < 995.0
    assert world.nodes[1].position.x > 989.0


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


def test_world_feeding_consumes_nutrients_from_field() -> None:
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

    assert world.nutrient_grid.sample(Vec2(100.0, 100.0)) < 2.0


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


def test_world_turns_dead_creatures_into_detritus() -> None:
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

    assert max(world.detritus_grid.values) > 0.0


def test_world_recycles_detritus_back_into_nutrients() -> None:
    config = Config.from_yaml(Path("config/default.yaml"))
    world = World(config=config)
    world.detritus_grid.set_value(col=0, row=0, value=4.0)
    nutrient_before = world.nutrient_grid.sample(Vec2(2.5, 2.5))

    world.step()

    assert world.detritus_grid.sample(Vec2(2.5, 2.5)) < 4.0
    assert world.nutrient_grid.sample(Vec2(2.5, 2.5)) > nutrient_before


def test_world_turbo_mode_skips_expensive_field_updates_between_full_ticks() -> None:
    config = Config.from_yaml(Path("config/default.yaml")).with_overrides(["environment.nutrient_source_count=0"])
    world = World(config=config, turbo=True)
    world.detritus_grid.set_value(col=0, row=0, value=4.0)
    nutrient_before = world.nutrient_grid.sample(Vec2(2.5, 2.5))

    world.step()

    assert world.detritus_grid.sample(Vec2(2.5, 2.5)) == 4.0
    assert world.nutrient_grid.sample(Vec2(2.5, 2.5)) == nutrient_before

    world.step(3)

    assert world.detritus_grid.sample(Vec2(2.5, 2.5)) < 4.0
    assert world.nutrient_grid.sample(Vec2(2.5, 2.5)) > nutrient_before


def test_world_reseeds_when_population_drops_below_minimum() -> None:
    config = Config.from_yaml(Path("config/default.yaml")).with_overrides(["creatures.min_population=3"])
    node = NodeState(
        position=Vec2(10.0, 10.0),
        velocity=Vec2.zero(),
        accumulated_force=Vec2.zero(),
        drag_coeff=1.0,
        radius=1.0,
    )
    creature = CreatureState(node_indices=(0,), energy=0.0005)
    world = World(config=config, nodes=[node], creatures=[creature], seed=7)

    world.step()

    assert len(world.creatures) >= 3


def test_world_can_return_top_creatures_by_energy() -> None:
    config = Config.from_yaml(Path("config/default.yaml"))
    nodes = [
        NodeState(
            position=Vec2(10.0, 10.0),
            velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(),
            drag_coeff=1.0,
            radius=1.0,
        ),
        NodeState(
            position=Vec2(20.0, 20.0),
            velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(),
            drag_coeff=1.0,
            radius=1.0,
        ),
    ]
    creatures = [
        CreatureState(node_indices=(0,), energy=1.0, id=10),
        CreatureState(node_indices=(1,), energy=2.0, id=20),
    ]
    world = World(config=config, nodes=nodes, creatures=creatures)

    top = world.get_top_creatures(n=1)

    assert [creature.id for creature in top] == [20]


def test_world_can_seed_demo_archetypes() -> None:
    config = Config.from_yaml(Path("config/default.yaml"))
    world = World(config=config, seed=7)

    world.seed_demo_archetypes()

    assert len(world.creatures) == 3
    assert len(world.nodes) >= 5
    assert len(world.edges) >= 2
    assert sum(1 for creature in world.creatures if creature.brain is not None) >= 2


def test_world_can_seed_from_exported_genomes(tmp_path: Path) -> None:
    config = Config.from_yaml(Path("config/default.yaml"))
    source_world = World(config=config, seed=7)
    source_world.seed_demo_archetypes()
    export_path = tmp_path / "top.json"
    source_world.export_top_creatures(path=export_path, n=2)

    target_world = World(config=config, seed=7)
    target_world.seed_from_exported_genomes(export_path)

    assert len(target_world.creatures) == 2
    assert all(creature.genome is not None for creature in target_world.creatures)


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
    assert any(creature.last_brain_outputs for creature in world.creatures if creature.brain is not None)


def test_world_reproduces_energy_rich_creatures() -> None:
    config = Config.from_yaml(Path("config/default.yaml"))
    brain = BrainState(
        input_weights=((1.0, 0.0, 0.0),),
        recurrent_weights=((0.5,),),
        biases=(0.0,),
        time_constants=(1.0,),
        states=(0.0,),
        output_size=1,
    )
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
    edges = [EdgeState(a=0, b=1, rest_length=6.0, stiffness=1.0, has_motor=True, motor_strength=2.0)]
    creatures = [CreatureState(node_indices=(0, 1), energy=200.0, brain=brain)]
    world = World(config=config, nodes=nodes, edges=edges, creatures=creatures)

    world.step()

    assert len(world.creatures) == 2
    assert len(world.nodes) == 4
    assert len(world.edges) == 2
    assert world.creatures[0].energy == world.creatures[1].energy
    assert world.nodes[2].radius != world.nodes[0].radius
    assert world.creatures[1].brain is not None
    assert world.creatures[1].brain.input_weights != world.creatures[0].brain.input_weights
    assert world.edges[1].motor_strength != world.edges[0].motor_strength
    assert world.creatures[0].genome is not None
    assert world.creatures[1].genome is not None
    decoded_nodes, decoded_edges, decoded_brain = decode_genome(
        genome=world.creatures[1].genome,
        anchor_position=world.nodes[2].position,
        drag_coeff=1.0,
    )
    assert decoded_nodes[0].radius == world.nodes[2].radius
    assert decoded_edges[0].motor_strength == world.edges[1].motor_strength
    assert decoded_brain is not None
    assert decoded_brain.input_weights == world.creatures[1].brain.input_weights


def test_world_stats_report_population_nodes_and_total_energy() -> None:
    config = Config.from_yaml(Path("config/default.yaml"))
    world = World(config=config, seed=7)
    world.seed_demo_archetypes()

    stats = world.stats()

    assert stats.population == 3
    assert stats.node_count == len(world.nodes)
    assert stats.edge_count == len(world.edges)
    assert stats.total_energy > 0.0
    assert stats.births == 3
    assert stats.deaths == 0
    assert stats.reproductions == 0


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

    assert result.stdout.strip() == "tick=3 seed=11 population=0 nodes=0 total_energy=0.000 births=0 deaths=0 reproductions=0"


def test_cli_run_command_can_seed_from_exported_genomes(tmp_path: Path) -> None:
    config = Config.from_yaml(Path("config/default.yaml"))
    source_world = World(config=config, seed=7)
    source_world.seed_demo_archetypes()
    export_path = tmp_path / "top.json"
    source_world.export_top_creatures(path=export_path, n=2)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "animalcula.cli",
            "run",
            "--config",
            "config/default.yaml",
            "--ticks",
            "0",
            "--seed",
            "11",
            "--seed-from",
            str(export_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "population=2" in result.stdout


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
    assert "births=3" in result.stdout


def test_cli_run_command_can_save_checkpoint(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "saved-world.json"
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
            "--save",
            str(checkpoint_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert checkpoint_path.exists()
    assert "tick=1" in result.stdout


def test_cli_run_command_can_resume_checkpoint(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "saved-world.json"
    subprocess.run(
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
            "--save",
            str(checkpoint_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "animalcula.cli",
            "run",
            "--resume",
            str(checkpoint_path),
            "--ticks",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "tick=3" in result.stdout
    assert "births=" in result.stdout


def test_cli_report_command_reads_checkpoint(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "saved-world.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "animalcula.cli",
            "run",
            "--config",
            "config/default.yaml",
            "--ticks",
            "2",
            "--seed",
            "11",
            "--seed-demo",
            "--save",
            str(checkpoint_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "animalcula.cli",
            "report",
            str(checkpoint_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "tick=2" in result.stdout
    assert "population=3" in result.stdout
    assert "births=3" in result.stdout


def test_cli_events_command_reads_checkpoint_events(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "saved-world.json"
    subprocess.run(
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
            "--save",
            str(checkpoint_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "animalcula.cli",
            "events",
            str(checkpoint_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "\"event_type\": \"birth\"" in result.stdout


def test_cli_run_command_can_log_periodic_stats(tmp_path: Path) -> None:
    log_path = tmp_path / "stats.jsonl"
    subprocess.run(
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
            "--seed-demo",
            "--log-stats",
            str(log_path),
            "--log-every",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert "\"tick\": 1" in lines[0]
    assert "\"tick\": 3" in lines[-1]


def test_cli_nursery_command_runs_and_saves_checkpoint(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "nursery.json"
    top_path = tmp_path / "top.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "animalcula.cli",
            "nursery",
            "--ticks",
            "5",
            "--seed",
            "11",
            "--top",
            "2",
            "--save-top",
            str(top_path),
            "--out",
            str(checkpoint_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert checkpoint_path.exists()
    assert top_path.exists()
    exported = json.loads(top_path.read_text(encoding="utf-8"))
    assert len(exported) == 2
    assert "top_creatures=" in result.stdout
    assert "saved=" in result.stdout


def test_cli_run_command_accepts_config_overrides() -> None:
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
            "--set",
            "energy.reproduction_threshold=0.1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "population=6" in result.stdout


def test_cli_run_command_accepts_turbo_mode() -> None:
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
            "--turbo",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "tick=1" in result.stdout
