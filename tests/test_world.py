from dataclasses import replace
import json
import sqlite3
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
    assert world.stats().peak_population == 0
    assert world.stats().population_variance == 0.0
    assert world.stats().population_capacity_fraction == 0.0
    assert world.stats().peak_population_capacity_fraction == 0.0
    assert world.stats().crowding_multiplier == 1.0
    assert world.stats().peak_crowding_multiplier == 1.0
    assert snapshot.phase_trace == [
        "environment",
        "sensing",
        "brain",
        "physics",
        "energy",
        "lifecycle",
    ]
    assert snapshot.world_width == config.world.width
    assert snapshot.world_height == config.world.height
    assert snapshot.total_energy == 0.0
    assert snapshot.nodes == ()
    assert snapshot.edges == ()
    assert snapshot.creatures == ()


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


def test_world_drag_regime_shift_changes_physics_multiplier() -> None:
    config = Config.from_yaml(Path("config/default.yaml")).with_overrides(
        [
            "environment.drag_shift_interval=2",
            "environment.drag_shift_multipliers=[1.0, 2.0]",
        ]
    )
    node = NodeState(
        position=Vec2.zero(),
        velocity=Vec2.zero(),
        accumulated_force=Vec2(2.0, 0.0),
        drag_coeff=1.0,
        radius=1.0,
    )
    world = World(config=config, nodes=[node])

    assert world.current_drag_multiplier() == 1.0
    world.step(1)
    assert world.nodes[0].position == Vec2(0.02, 0.0)
    assert world.current_drag_multiplier() == 2.0

    world.nodes[0] = replace(world.nodes[0], accumulated_force=Vec2(2.0, 0.0))
    world.step(1)
    assert world.nodes[0].position == Vec2(0.03, 0.0)


def test_world_nutrient_epoch_reseeds_sources_and_changes_strength_multiplier() -> None:
    config = Config.from_yaml(Path("config/default.yaml")).with_overrides(
        [
            "environment.nutrient_source_count=3",
            "environment.nutrient_shift_interval=0",
            "environment.nutrient_epoch_interval=2",
            "environment.nutrient_epoch_strength_multipliers=[1.0, 0.5, 1.5]",
        ]
    )
    world = World(config=config, seed=7)
    before = list(world._nutrient_source_cells)

    assert world.current_nutrient_source_strength_multiplier() == 1.0

    world.step()
    after_one = list(world._nutrient_source_cells)

    world.step()
    after_two = list(world._nutrient_source_cells)

    assert after_one == before
    assert after_two != before
    assert world.current_nutrient_source_strength_multiplier() == 0.5
    assert world.stats().nutrient_source_strength_multiplier == 0.5
    assert world.nutrient_grid.sample(world.nutrient_grid.position_for_cell(*after_two[0])) == 1.0


def test_world_nutrient_epoch_reseeds_are_deterministic_for_same_seed() -> None:
    config = Config.from_yaml(Path("config/default.yaml")).with_overrides(
        [
            "environment.nutrient_source_count=4",
            "environment.nutrient_shift_interval=0",
            "environment.nutrient_epoch_interval=2",
            "environment.nutrient_epoch_strength_multipliers=[1.0, 0.5]",
        ]
    )
    world_a = World(config=config, seed=7)
    world_b = World(config=config, seed=7)

    world_a.step(4)
    world_b.step(4)

    assert world_a._nutrient_source_cells == world_b._nutrient_source_cells
    assert world_a.current_nutrient_source_strength_multiplier() == world_b.current_nutrient_source_strength_multiplier()


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


def test_world_snapshot_contains_renderable_creature_graph() -> None:
    config = Config.from_yaml(Path("config/default.yaml"))
    world = World(config=config, seed=7)
    world.seed_demo_archetypes()

    snapshot = world.snapshot()

    assert snapshot.population == 3
    assert len(snapshot.nodes) == len(world.nodes)
    assert len(snapshot.edges) == len(world.edges)
    assert len(snapshot.creatures) == len(world.creatures)
    assert all(node.creature_id is not None for node in snapshot.nodes)
    assert {creature.trophic_role for creature in snapshot.creatures} >= {"autotroph", "herbivore", "predator"}


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
        NodeState(
            position=Vec2(200.0, 200.0),
            velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(),
            drag_coeff=1.0,
            radius=1.0,
            node_type=NodeType.SENSOR,
        ),
    ]
    creature = CreatureState(node_indices=(0, 1, 2), energy=1.0)
    world = World(config=config, nodes=nodes, creatures=[creature])
    world.nutrient_grid.set_value(col=19, row=20, value=1.0)
    world.nutrient_grid.set_value(col=21, row=20, value=3.0)
    world.chemical_a_grid.set_value(col=39, row=40, value=1.0)
    world.chemical_a_grid.set_value(col=41, row=40, value=3.0)
    world.chemical_b_grid.set_value(col=40, row=40, value=2.0)

    world.step()

    sensed = world.creatures[0].last_sensed_inputs
    assert len(sensed) == 16
    assert sensed[3] > 0.0
    assert sensed[5] > 0.0
    assert sensed[7] > 0.0
    assert sensed[8] > 0.0
    assert sensed[9] > 0.0
    assert sensed[10] > 0.0


def test_world_brain_outputs_can_emit_chemicals() -> None:
    config = Config.from_yaml(Path("config/default.yaml"))
    brain = BrainState(
        input_weights=(
            (0.0,) * 16,
            (0.0,) * 16,
            (0.0,) * 16,
            (0.0,) * 16,
        ),
        recurrent_weights=(
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0),
        ),
        biases=(0.0, 0.0, 10.0, 10.0),
        time_constants=(1.0, 1.0, 1.0, 1.0),
        states=(0.0, 0.0, 0.0, 0.0),
        output_size=4,
    )
    nodes = [
        NodeState(
            position=Vec2(100.0, 100.0),
            velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(),
            drag_coeff=1.0,
            radius=1.0,
            node_type=NodeType.MOUTH,
        ),
    ]
    creature = CreatureState(node_indices=(0,), energy=1.0, brain=brain)
    world = World(config=config, nodes=nodes, creatures=[creature])

    world.step()

    assert world.chemical_a_grid.sample(Vec2(100.0, 100.0)) > 0.0
    assert world.chemical_b_grid.sample(Vec2(100.0, 100.0)) > 0.0


def test_world_can_latch_grippers_when_outputs_are_active() -> None:
    config = Config.from_yaml(Path("config/default.yaml")).with_overrides(
        [
            "energy.basal_cost_per_node=0.0",
            "energy.feed_rate=0.0",
            "energy.photosynthesis_rate=0.0",
            "energy.grip_cost=0.0",
        ]
    )
    grip_brain = BrainState(
        input_weights=((0.0,) * 16,),
        recurrent_weights=((0.0,),),
        biases=(10.0,),
        time_constants=(1.0,),
        states=(0.0,),
        output_size=1,
    )
    nodes = [
        NodeState(
            position=Vec2(100.0, 100.0),
            velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(),
            drag_coeff=1.0,
            radius=1.0,
            node_type=NodeType.GRIPPER,
        ),
        NodeState(
            position=Vec2(101.5, 100.0),
            velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(),
            drag_coeff=1.0,
            radius=1.0,
            node_type=NodeType.GRIPPER,
        ),
    ]
    creatures = [
        CreatureState(node_indices=(0,), energy=1.0, brain=grip_brain),
        CreatureState(node_indices=(1,), energy=1.0, brain=grip_brain),
    ]
    world = World(config=config, nodes=nodes, creatures=creatures)

    world.step()

    assert len(world.grip_latches) == 1


def test_world_can_latch_active_gripper_to_overlapping_prey_body() -> None:
    config = Config.from_yaml(Path("config/default.yaml")).with_overrides(
        [
            "energy.basal_cost_per_node=0.0",
            "energy.feed_rate=0.0",
            "energy.photosynthesis_rate=0.0",
            "energy.grip_cost=0.0",
        ]
    )
    grip_brain = BrainState(
        input_weights=((0.0,) * 16,),
        recurrent_weights=((0.0,),),
        biases=(10.0,),
        time_constants=(1.0,),
        states=(0.0,),
        output_size=1,
    )
    nodes = [
        NodeState(
            position=Vec2(100.0, 100.0),
            velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(),
            drag_coeff=1.0,
            radius=1.0,
            node_type=NodeType.GRIPPER,
        ),
        NodeState(
            position=Vec2(101.5, 100.0),
            velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(),
            drag_coeff=1.0,
            radius=1.0,
        ),
    ]
    creatures = [
        CreatureState(node_indices=(0,), energy=1.0, brain=grip_brain),
        CreatureState(node_indices=(1,), energy=1.0),
    ]
    world = World(config=config, nodes=nodes, creatures=creatures)

    world.step()

    assert len(world.grip_latches) == 1
    assert world.grip_latches[0].node_a_index == 0
    assert world.grip_latches[0].node_b_index == 1


def test_world_can_latch_active_gripper_to_nearby_prey_body() -> None:
    config = Config.from_yaml(Path("config/default.yaml")).with_overrides(
        [
            "energy.basal_cost_per_node=0.0",
            "energy.feed_rate=0.0",
            "energy.photosynthesis_rate=0.0",
            "energy.grip_cost=0.0",
        ]
    )
    grip_brain = BrainState(
        input_weights=((0.0,) * 16,),
        recurrent_weights=((0.0,),),
        biases=(10.0,),
        time_constants=(1.0,),
        states=(0.0,),
        output_size=1,
    )
    nodes = [
        NodeState(
            position=Vec2(100.0, 100.0),
            velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(),
            drag_coeff=1.0,
            radius=1.0,
            node_type=NodeType.GRIPPER,
        ),
        NodeState(
            position=Vec2(102.5, 100.0),
            velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(),
            drag_coeff=1.0,
            radius=1.0,
        ),
    ]
    creatures = [
        CreatureState(node_indices=(0,), energy=1.0, brain=grip_brain),
        CreatureState(node_indices=(1,), energy=1.0),
    ]
    world = World(config=config, nodes=nodes, creatures=creatures)

    world.step()

    assert len(world.grip_latches) == 1


def test_world_releases_grippers_when_outputs_drop_inactive() -> None:
    config = Config.from_yaml(Path("config/default.yaml")).with_overrides(
        [
            "energy.basal_cost_per_node=0.0",
            "energy.feed_rate=0.0",
            "energy.photosynthesis_rate=0.0",
            "energy.grip_cost=0.0",
        ]
    )
    active_brain = BrainState(
        input_weights=((0.0,) * 16,),
        recurrent_weights=((0.0,),),
        biases=(10.0,),
        time_constants=(1.0,),
        states=(0.0,),
        output_size=1,
    )
    inactive_brain = BrainState(
        input_weights=((0.0,) * 16,),
        recurrent_weights=((0.0,),),
        biases=(-10.0,),
        time_constants=(1.0,),
        states=(0.0,),
        output_size=1,
    )
    nodes = [
        NodeState(
            position=Vec2(100.0, 100.0),
            velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(),
            drag_coeff=1.0,
            radius=1.0,
            node_type=NodeType.GRIPPER,
        ),
        NodeState(
            position=Vec2(101.5, 100.0),
            velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(),
            drag_coeff=1.0,
            radius=1.0,
            node_type=NodeType.GRIPPER,
        ),
    ]
    creatures = [
        CreatureState(node_indices=(0,), energy=1.0, brain=active_brain),
        CreatureState(node_indices=(1,), energy=1.0, brain=active_brain),
    ]
    world = World(config=config, nodes=nodes, creatures=creatures)

    world.step()
    world.creatures = [
        CreatureState(
            node_indices=creature.node_indices,
            energy=creature.energy,
            brain=inactive_brain,
            genome=creature.genome,
            mean_speed_recent=creature.mean_speed_recent,
            last_sensed_inputs=creature.last_sensed_inputs,
            last_brain_outputs=creature.last_brain_outputs,
            id=creature.id,
            parent_id=creature.parent_id,
            age_ticks=creature.age_ticks,
        )
        for creature in world.creatures
    ]

    world.step()

    assert world.grip_latches == []


def test_world_exposes_gripper_contact_and_active_signals() -> None:
    config = Config.from_yaml(Path("config/default.yaml")).with_overrides(
        [
            "energy.basal_cost_per_node=0.0",
            "energy.feed_rate=0.0",
            "energy.photosynthesis_rate=0.0",
            "energy.grip_cost=0.0",
        ]
    )
    grip_brain = BrainState(
        input_weights=((0.0,) * 16,),
        recurrent_weights=((0.0,),),
        biases=(10.0,),
        time_constants=(1.0,),
        states=(0.0,),
        output_size=1,
    )
    nodes = [
        NodeState(
            position=Vec2(100.0, 100.0),
            velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(),
            drag_coeff=1.0,
            radius=1.0,
            node_type=NodeType.GRIPPER,
        ),
        NodeState(
            position=Vec2(101.5, 100.0),
            velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(),
            drag_coeff=1.0,
            radius=1.0,
            node_type=NodeType.GRIPPER,
        ),
    ]
    creatures = [
        CreatureState(node_indices=(0,), energy=1.0, brain=grip_brain),
        CreatureState(node_indices=(1,), energy=1.0, brain=grip_brain),
    ]
    world = World(config=config, nodes=nodes, creatures=creatures)

    world.step()
    world.step()

    assert world.creatures[0].last_sensed_inputs[-2] > 0.0
    assert world.creatures[0].last_sensed_inputs[-1] > 0.0
    assert world.creatures[1].last_sensed_inputs[-2] > 0.0
    assert world.creatures[1].last_sensed_inputs[-1] > 0.0


def test_world_charges_energy_for_active_grips() -> None:
    base_config = Config.from_yaml(Path("config/default.yaml")).with_overrides(
        [
            "energy.basal_cost_per_node=0.0",
            "energy.feed_rate=0.0",
            "energy.photosynthesis_rate=0.0",
            "energy.grip_cost=0.0",
        ]
    )
    costly_config = base_config.with_overrides(["energy.grip_cost=0.05"])
    grip_brain = BrainState(
        input_weights=((0.0,) * 16,),
        recurrent_weights=((0.0,),),
        biases=(10.0,),
        time_constants=(1.0,),
        states=(0.0,),
        output_size=1,
    )
    nodes = [
        NodeState(
            position=Vec2(100.0, 100.0),
            velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(),
            drag_coeff=1.0,
            radius=1.0,
            node_type=NodeType.GRIPPER,
        ),
        NodeState(
            position=Vec2(101.5, 100.0),
            velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(),
            drag_coeff=1.0,
            radius=1.0,
            node_type=NodeType.GRIPPER,
        ),
    ]
    creatures = [
        CreatureState(node_indices=(0,), energy=1.0, brain=grip_brain),
        CreatureState(node_indices=(1,), energy=1.0, brain=grip_brain),
    ]
    base_world = World(config=base_config, nodes=nodes, creatures=creatures)
    costly_world = World(config=costly_config, nodes=nodes, creatures=creatures)

    base_world.step()
    costly_world.step()

    assert costly_world.creatures[0].energy < base_world.creatures[0].energy
    assert costly_world.creatures[1].energy < base_world.creatures[1].energy


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
    assert world.stats().crowding_multiplier == 2.0
    assert world.stats().peak_crowding_multiplier == 2.0


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


def test_world_charges_energy_for_motor_actuation() -> None:
    config = Config.from_yaml(Path("config/default.yaml")).with_overrides(["energy.motor_cost_per_unit=0.0"])
    costly_config = config.with_overrides(["energy.motor_cost_per_unit=0.1"])
    brain = BrainState(
        input_weights=((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),),
        recurrent_weights=((0.0,),),
        biases=(10.0,),
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
    creature = CreatureState(node_indices=(0, 1), energy=2.0, brain=brain)
    world = World(config=config, nodes=nodes, edges=edges, creatures=[creature])
    costly_world = World(config=costly_config, nodes=nodes, edges=edges, creatures=[creature])

    world.step()
    costly_world.step()

    assert costly_world.creatures[0].energy < world.creatures[0].energy


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


def test_world_can_scavenge_energy_from_detritus() -> None:
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
    world.detritus_grid.set_value(col=20, row=20, value=2.0)

    world.step()

    assert world.creatures[0].energy > 1.0
    assert world.detritus_grid.sample(Vec2(100.0, 100.0)) < 2.0


def test_world_can_predate_overlapping_creature_with_bite_output() -> None:
    config = Config.from_yaml(Path("config/default.yaml")).with_overrides(
        [
            "energy.basal_cost_per_node=0.0",
            "energy.feed_rate=0.0",
            "energy.scavenging_rate=0.0",
            "energy.photosynthesis_rate=0.0",
            "energy.predation_rate=0.5",
            "energy.predation_transfer_efficiency=1.0",
        ]
    )
    predator_brain = BrainState(
        input_weights=(
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ),
        recurrent_weights=((0.0, 0.0), (0.0, 0.0)),
        biases=(10.0, 10.0),
        time_constants=(1.0, 1.0),
        states=(0.0, 0.0),
        output_size=2,
    )
    nodes = [
        NodeState(
            position=Vec2(100.0, 100.0),
            velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(),
            drag_coeff=1.0,
            radius=1.0,
            node_type=NodeType.MOUTH,
        ),
        NodeState(
            position=Vec2(100.0, 100.5),
            velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(),
            drag_coeff=1.0,
            radius=1.0,
            node_type=NodeType.GRIPPER,
        ),
        NodeState(
            position=Vec2(100.5, 100.0),
            velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(),
            drag_coeff=1.0,
            radius=1.0,
        ),
    ]
    creatures = [
        CreatureState(node_indices=(0, 1), energy=1.0, brain=predator_brain),
        CreatureState(node_indices=(2,), energy=1.0),
    ]
    world = World(config=config, nodes=nodes, creatures=creatures)

    world.step()

    assert world.creatures[0].energy > 1.0
    assert world.creatures[1].energy < 1.0


def test_world_can_predate_gripped_creature_without_direct_mouth_overlap() -> None:
    config = Config.from_yaml(Path("config/default.yaml")).with_overrides(
        [
            "energy.basal_cost_per_node=0.0",
            "energy.feed_rate=0.0",
            "energy.scavenging_rate=0.0",
            "energy.photosynthesis_rate=0.0",
            "energy.grip_cost=0.0",
            "energy.predation_rate=0.5",
            "energy.predation_transfer_efficiency=1.0",
        ]
    )
    predator_brain = BrainState(
        input_weights=(
            (0.0,) * 16,
            (0.0,) * 16,
        ),
        recurrent_weights=((0.0, 0.0), (0.0, 0.0)),
        biases=(10.0, 10.0),
        time_constants=(1.0, 1.0),
        states=(0.0, 0.0),
        output_size=2,
    )
    nodes = [
        NodeState(
            position=Vec2(100.0, 100.0),
            velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(),
            drag_coeff=1.0,
            radius=1.0,
            node_type=NodeType.MOUTH,
        ),
        NodeState(
            position=Vec2(100.0, 102.0),
            velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(),
            drag_coeff=1.0,
            radius=1.0,
            node_type=NodeType.GRIPPER,
        ),
        NodeState(
            position=Vec2(100.0, 103.0),
            velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(),
            drag_coeff=1.0,
            radius=1.0,
        ),
    ]
    creatures = [
        CreatureState(node_indices=(0, 1), energy=1.0, brain=predator_brain),
        CreatureState(node_indices=(2,), energy=1.0),
    ]
    world = World(config=config, nodes=nodes, creatures=creatures)

    world.step()

    assert world.grip_latches != []
    assert world.creatures[0].energy > 1.0
    assert world.creatures[1].energy < 1.0


def test_world_does_not_predate_with_mouth_only_bite_output() -> None:
    config = Config.from_yaml(Path("config/default.yaml")).with_overrides(
        [
            "energy.basal_cost_per_node=0.0",
            "energy.feed_rate=0.0",
            "energy.scavenging_rate=0.0",
            "energy.photosynthesis_rate=0.0",
            "energy.predation_rate=0.5",
            "energy.predation_transfer_efficiency=1.0",
        ]
    )
    predator_brain = BrainState(
        input_weights=((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),),
        recurrent_weights=((0.0,),),
        biases=(10.0,),
        time_constants=(1.0,),
        states=(0.0,),
        output_size=1,
    )
    nodes = [
        NodeState(
            position=Vec2(100.0, 100.0),
            velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(),
            drag_coeff=1.0,
            radius=1.0,
            node_type=NodeType.MOUTH,
        ),
        NodeState(
            position=Vec2(100.5, 100.0),
            velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(),
            drag_coeff=1.0,
            radius=1.0,
        ),
    ]
    creatures = [
        CreatureState(node_indices=(0,), energy=1.0, brain=predator_brain),
        CreatureState(node_indices=(1,), energy=1.0),
    ]
    world = World(config=config, nodes=nodes, creatures=creatures)

    world.step()

    assert world.creatures[0].energy <= 1.0
    assert world.creatures[1].energy == 1.0


def test_world_classifies_mouth_only_biter_as_herbivore() -> None:
    config = Config.from_yaml(Path("config/default.yaml"))
    brain = BrainState(
        input_weights=((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),),
        recurrent_weights=((0.0,),),
        biases=(0.0,),
        time_constants=(1.0,),
        states=(0.0,),
        output_size=1,
    )
    node = NodeState(
        position=Vec2(100.0, 100.0),
        velocity=Vec2.zero(),
        accumulated_force=Vec2.zero(),
        drag_coeff=1.0,
        radius=1.0,
        node_type=NodeType.MOUTH,
    )
    creature = CreatureState(node_indices=(0,), energy=1.0, brain=brain)
    world = World(config=config, nodes=[node], creatures=[creature])

    assert world._trophic_role(world.creatures[0]) == "herbivore"


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


def test_world_shifts_nutrient_sources_when_interval_is_reached() -> None:
    config = Config.from_yaml(Path("config/default.yaml")).with_overrides(
        [
            "environment.nutrient_source_count=3",
            "environment.nutrient_shift_interval=2",
            "environment.nutrient_shift_count=1",
        ]
    )
    world = World(config=config, seed=7)
    before = list(world._nutrient_source_cells)

    world.step()
    after_one = list(world._nutrient_source_cells)
    world.step()
    after_two = list(world._nutrient_source_cells)

    assert after_one == before
    assert after_two != before
    assert len(after_two) == len(before)
    assert len(set(after_two)) == len(after_two)


def test_world_nutrient_source_shifts_are_deterministic_for_same_seed() -> None:
    config = Config.from_yaml(Path("config/default.yaml")).with_overrides(
        [
            "environment.nutrient_source_count=4",
            "environment.nutrient_shift_interval=2",
            "environment.nutrient_shift_count=2",
        ]
    )
    world_a = World(config=config, seed=7)
    world_b = World(config=config, seed=7)

    world_a.step(4)
    world_b.step(4)

    assert world_a._nutrient_source_cells == world_b._nutrient_source_cells


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
    assert any(creature.brain is not None and creature.brain.output_size >= 3 for creature in world.creatures)


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


def test_world_reproduction_can_grow_child_hidden_neuron_capacity() -> None:
    config = Config.from_yaml(Path("config/default.yaml")).with_overrides(
        [
            "evolution.position_mutation_sigma=0.0",
            "evolution.radius_mutation_sigma=0.0",
            "evolution.weight_mutation_sigma=0.0",
            "evolution.bias_mutation_sigma=0.0",
            "evolution.tau_mutation_sigma=0.0",
            "evolution.motor_strength_mutation_sigma=0.0",
            "evolution.motor_toggle_mutation_rate=0.0",
            "evolution.node_type_mutation_rate=0.0",
            "evolution.structural_mutation_rate=0.0",
            "evolution.hidden_neuron_mutation_rate=1.0",
            "evolution.max_hidden_neurons=24",
        ]
    )
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
    world = World(config=config, nodes=nodes, edges=edges, creatures=creatures, seed=7)

    world.step()

    assert len(world.creatures) == 2
    assert world.creatures[1].brain is not None
    assert len(world.creatures[1].brain.biases) == len(world.creatures[0].brain.biases) + 1
    assert world.creatures[1].brain.output_size == world.creatures[0].brain.output_size


def test_world_requires_reproduce_signal_when_brain_exposes_one() -> None:
    config = Config.from_yaml(Path("config/default.yaml"))
    brain = BrainState(
        input_weights=((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
        recurrent_weights=((0.0, 0.0), (0.0, 0.0)),
        biases=(0.0, -10.0),
        time_constants=(1.0, 1.0),
        states=(0.0, 0.0),
        output_size=2,
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

    assert len(world.creatures) == 1


def test_world_reproduces_when_reproduce_signal_is_high() -> None:
    config = Config.from_yaml(Path("config/default.yaml"))
    brain = BrainState(
        input_weights=((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
        recurrent_weights=((0.0, 0.0), (0.0, 0.0)),
        biases=(0.0, 10.0),
        time_constants=(1.0, 1.0),
        states=(0.0, 0.0),
        output_size=2,
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
    assert stats.speciation_events == 0
    assert stats.species_extinctions == 0
    assert stats.species_turnover == 0
    assert stats.longest_species_lifespan == 0
    assert stats.mean_extinct_species_lifespan == 0.0
    assert stats.lineage_count == 3
    assert stats.species_count == 3
    assert stats.observed_species_count == 3
    assert stats.peak_species_count == 3
    assert stats.diversity_index > 0.0
    assert stats.mean_nodes_per_creature > 0.0
    assert stats.autotroph_count >= 1
    assert stats.herbivore_count >= 1
    assert stats.predator_count >= 1


def test_demo_seed_includes_spec_aligned_triangle_predator() -> None:
    config = Config.from_yaml(Path("config/default.yaml"))
    world = World(config=config, seed=7)
    world.seed_demo_archetypes()
    primary_nutrient_source = world.nutrient_grid.position_for_cell(*world._nutrient_source_cells[0])

    predator = next(
        creature
        for creature in world.creatures
        if any(world.nodes[node_index].node_type == NodeType.GRIPPER for node_index in creature.node_indices)
    )

    predator_nodes = [world.nodes[node_index] for node_index in predator.node_indices]
    internal_edges = [
        edge
        for edge in world.edges
        if edge.a in predator.node_indices and edge.b in predator.node_indices
    ]

    assert len(predator_nodes) == 3
    assert sum(1 for node in predator_nodes if node.node_type == NodeType.MOUTH) == 1
    assert sum(1 for node in predator_nodes if node.node_type == NodeType.GRIPPER) == 1
    assert sum(1 for node in predator_nodes if node.node_type == NodeType.PHOTORECEPTOR) == 0
    assert len(internal_edges) == 3
    assert next(node.position for node in predator_nodes if node.node_type == NodeType.MOUTH) == primary_nutrient_source


def test_world_can_build_species_snapshots() -> None:
    config = Config.from_yaml(Path("config/default.yaml"))
    world = World(config=config, seed=7)
    world.seed_demo_archetypes()

    snapshots = world.species_snapshots()

    assert len(snapshots) == 3
    assert all(snapshot["count"] == 1 for snapshot in snapshots)
    assert all("mean_energy" in snapshot for snapshot in snapshots)


def test_world_can_build_phenotype_snapshots() -> None:
    config = Config.from_yaml(Path("config/default.yaml"))
    world = World(config=config, seed=7)
    world.seed_demo_archetypes()
    world.step()

    snapshots = world.phenotype_snapshots()

    assert len(snapshots) == 3
    assert all("num_nodes" in snapshot for snapshot in snapshots)
    assert all("mean_speed_recent" in snapshot for snapshot in snapshots)
    assert all("species_id" in snapshot for snapshot in snapshots)


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

    assert (
        result.stdout.strip()
        == "tick=3 seed=11 drag_multiplier=1.00 nutrient_strength_multiplier=1.00 population=0 peak_population=0 population_variance=0.000 population_capacity_fraction=0.000 peak_population_capacity_fraction=0.000 crowding_multiplier=1.000 peak_crowding_multiplier=1.000 nodes=0 total_energy=0.000 births=0 deaths=0 reproductions=0 speciations=0 species_extinctions=0 species_turnover=0 predation_kills=0 environment_perturbations=0 species=0 observed_species=0 peak_species=0 peak_species_fraction=0.000 lineages=0 runaway_dominance=false diversity=0.000 complexity=0.00 longest_species_lifespan=0 mean_extinct_species_lifespan=0.00 autotrophs=0 herbivores=0 predators=0 trophic_balance=0.000"
    )


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
    assert "speciations=0" in result.stdout
    assert "species_extinctions=0" in result.stdout
    assert "species_turnover=0" in result.stdout
    assert "species=3" in result.stdout
    assert "observed_species=3" in result.stdout
    assert "peak_species=3" in result.stdout
    assert "lineages=3" in result.stdout
    assert "predation_kills=0" in result.stdout
    assert "complexity=" in result.stdout
    assert "mean_extinct_species_lifespan=" in result.stdout
    assert "predators=" in result.stdout


def test_cli_view_help_describes_minimal_debug_viewer() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "animalcula.cli",
            "view",
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "minimal local debug viewer" in result.stdout
    assert "--steps-per-frame" in result.stdout


def test_cli_species_command_reads_checkpoint_species_snapshots(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "species.json"
    world = World(config=Config.from_yaml(Path("config/default.yaml")), seed=7)
    world.seed_demo_archetypes()
    world.save(checkpoint_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "animalcula.cli",
            "species",
            str(checkpoint_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "\"species_id\": " in result.stdout
    assert "\"mean_energy\": " in result.stdout


def test_cli_phenotypes_command_reads_checkpoint_phenotypes(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "phenotypes.json"
    world = World(config=Config.from_yaml(Path("config/default.yaml")), seed=7)
    world.seed_demo_archetypes()
    world.step()
    world.save(checkpoint_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "animalcula.cli",
            "phenotypes",
            str(checkpoint_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "\"num_nodes\": " in result.stdout
    assert "\"mean_speed_recent\": " in result.stdout


def test_cli_extract_genomes_writes_top_creatures_from_checkpoint(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "extract-source.json"
    out_path = tmp_path / "top-creatures.json"
    world = World(config=Config.from_yaml(Path("config/default.yaml")), seed=7)
    world.seed_demo_archetypes()
    world.step(5)
    world.save(checkpoint_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "animalcula.cli",
            "extract-genomes",
            str(checkpoint_path),
            "--top",
            "2",
            "--out",
            str(out_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert len(payload) == 2
    assert payload[0]["genome"] is not None
    assert "saved=" in result.stdout


def test_cli_evaluate_genomes_ranks_seed_bank_and_exports_promoted_set(tmp_path: Path) -> None:
    source_world = World(config=Config.from_yaml(Path("config/default.yaml")), seed=7)
    source_world.seed_demo_archetypes()
    seed_bank_path = tmp_path / "seed-bank.json"
    out_path = tmp_path / "seed-report.json"
    promoted_path = tmp_path / "promoted.json"
    source_world.export_top_creatures(path=seed_bank_path, n=1)

    payload = json.loads(seed_bank_path.read_text(encoding="utf-8"))
    low_energy = {**payload[0], "energy": 0.1}
    high_energy = {**payload[0], "energy": 10.0}
    seed_bank_path.write_text(json.dumps([low_energy, high_energy], indent=2), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "animalcula.cli",
            "evaluate-genomes",
            str(seed_bank_path),
            "--config",
            "config/default.yaml",
            "--ticks",
            "1",
            "--seeds",
            "11,12",
            "--workers",
            "1",
            "--top",
            "1",
            "--save-top",
            str(promoted_path),
            "--out",
            str(out_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["candidate_count"] == 2
    assert report["seeds"] == [11, 12]
    assert len(report["rankings"]) == 2
    assert report["rankings"][0]["source_energy"] == 10.0
    assert "avg_species_turnover" in report["rankings"][0]
    assert "promoted_energy" in report["rankings"][0]

    promoted = json.loads(promoted_path.read_text(encoding="utf-8"))
    assert len(promoted) == 1
    assert promoted[0]["genome"] is not None
    assert promoted[0]["energy"] > 0.0
    assert "saved=" in result.stdout
    assert "evaluated=2" in result.stdout
    assert "promoted=" in result.stdout


def test_cli_promote_genomes_runs_multiple_rounds_and_writes_manifest(tmp_path: Path) -> None:
    source_world = World(config=Config.from_yaml(Path("config/default.yaml")), seed=7)
    source_world.seed_demo_archetypes()
    seed_bank_path = tmp_path / "seed-bank.json"
    out_dir = tmp_path / "promotion"
    source_world.export_top_creatures(path=seed_bank_path, n=2)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "animalcula.cli",
            "promote-genomes",
            str(seed_bank_path),
            "--config",
            "config/default.yaml",
            "--ticks",
            "1",
            "--seeds",
            "11",
            "--workers",
            "1",
            "--rounds",
            "2",
            "--top",
            "1",
            "--out-dir",
            str(out_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    manifest = json.loads((out_dir / "promotion.json").read_text(encoding="utf-8"))
    assert manifest["rounds_requested"] == 2
    assert manifest["rounds_completed"] == 2
    assert len(manifest["rounds"]) == 2
    assert "carryover_from_previous_round_count" in manifest["rounds"][1]
    assert "diversity_drift" in manifest["rounds"][0]
    assert "stable_top_rank_streak" in manifest
    assert Path(manifest["final_genomes_path"]).exists()
    assert "saved=" in result.stdout
    assert "rounds=2" in result.stdout
    assert "final=" in result.stdout


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
    assert "\"genome_hash\": \"" in result.stdout


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
    assert "\"peak_population\":" in lines[0]
    assert "\"population_variance\":" in lines[0]
    assert "\"population_capacity_fraction\":" in lines[0]
    assert "\"peak_population_capacity_fraction\":" in lines[0]
    assert "\"crowding_multiplier\":" in lines[0]
    assert "\"peak_crowding_multiplier\":" in lines[0]
    assert "\"drag_multiplier\": 1.0" in lines[0]
    assert "\"nutrient_source_strength_multiplier\": 1.0" in lines[0]
    assert "\"peak_species_fraction\":" in lines[0]
    assert "\"runaway_dominance_detected\":" in lines[0]
    assert "\"environment_perturbations\":" in lines[0]
    assert "\"trophic_balance_score\":" in lines[0]
    assert "\"tick\": 3" in lines[-1]


def test_cli_run_command_can_log_periodic_stats_to_sqlite(tmp_path: Path) -> None:
    log_path = tmp_path / "stats.sqlite"
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
            "--log-stats-sqlite",
            str(log_path),
            "--log-every",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    with sqlite3.connect(log_path) as connection:
        metadata = connection.execute(
            """
            SELECT seed, turbo, config_json
            FROM run_metadata
            WHERE id = 1
            """
        ).fetchone()
        rows = connection.execute(
            """
            SELECT tick, peak_population, crowding_multiplier, peak_crowding_multiplier,
                   drag_multiplier, nutrient_source_strength_multiplier,
                   environment_perturbations, trophic_balance_score
            FROM stats_log
            ORDER BY tick
            """
        ).fetchall()

    assert metadata is not None
    assert metadata[0] == 11
    assert metadata[1] == 0
    assert "\"max_population\": 500" in metadata[2]
    assert len(rows) == 3
    assert rows[0][0] == 1
    assert rows[0][1] >= 3
    assert rows[0][2] >= 1.0
    assert rows[0][3] >= 1.0
    assert rows[0][4] == 1.0
    assert rows[0][5] == 1.0
    assert rows[0][6] == 0
    assert rows[-1][0] == 3
    assert rows[0][7] >= 0.0


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

    assert "population=5" in result.stdout
    assert "reproductions=2" in result.stdout


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
