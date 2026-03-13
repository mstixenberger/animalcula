from pathlib import Path

from animalcula import Config, World
from animalcula.sim.types import BrainState, CreatureState, NodeState, NodeType, Vec2


def test_world_checkpoint_roundtrip_preserves_state(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "world.json"
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
    world = World(config=config, seed=7, nodes=nodes, creatures=creatures)
    world._runaway_dominance_tick_threshold = 1
    world.step(2)

    world.save(checkpoint_path)
    restored = World.load(checkpoint_path)

    assert restored.tick == world.tick
    assert restored.seed == world.seed
    assert restored.nodes == world.nodes
    assert restored.edges == world.edges
    assert restored.creatures == world.creatures
    assert restored.nutrient_grid.values == world.nutrient_grid.values
    assert restored.light_grid.values == world.light_grid.values
    assert restored.chemical_a_grid.values == world.chemical_a_grid.values
    assert restored.chemical_b_grid.values == world.chemical_b_grid.values
    assert restored.grip_latches == world.grip_latches
    assert restored.events == world.events
    assert restored.stats().peak_population == world.stats().peak_population
    assert restored.stats().population_variance == world.stats().population_variance
    assert restored.stats().peak_species_fraction == world.stats().peak_species_fraction
    assert restored.stats().peak_grip_latch_count == world.stats().peak_grip_latch_count
    assert restored.stats().runaway_dominance_detected == world.stats().runaway_dominance_detected
