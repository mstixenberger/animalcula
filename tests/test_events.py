from pathlib import Path

from animalcula import Config, World
from animalcula.sim.types import BrainState, CreatureState, EdgeState, NodeState, NodeType, Vec2


def test_world_logs_reproduction_and_death_events() -> None:
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
        NodeState(
            position=Vec2(10.0, 10.0),
            velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(),
            drag_coeff=1.0,
            radius=1.0,
        ),
    ]
    edges = [EdgeState(a=0, b=1, rest_length=6.0, stiffness=1.0, has_motor=True, motor_strength=2.0)]
    creatures = [
        CreatureState(node_indices=(0, 1), energy=200.0),
        CreatureState(node_indices=(2,), energy=0.0001),
    ]
    world = World(config=config, nodes=nodes, edges=edges, creatures=creatures)
    world.creatures[0] = CreatureState(
        node_indices=world.creatures[0].node_indices,
        energy=world.creatures[0].energy,
        brain=brain,
        id=world.creatures[0].id,
        parent_id=world.creatures[0].parent_id,
    )

    world.step()

    event_types = [event.event_type for event in world.events]
    assert "reproduction" in event_types
    assert "birth" in event_types
    assert "death" in event_types
    assert all(event.genome_hash for event in world.events)


def test_seeded_creatures_receive_unique_ids() -> None:
    config = Config.from_yaml(Path("config/default.yaml"))
    world = World(config=config, seed=7)
    world.seed_demo_archetypes()

    ids = [creature.id for creature in world.creatures]

    assert len(ids) == len(set(ids))
