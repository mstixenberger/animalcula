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


def test_world_logs_predation_kill_events() -> None:
    config = Config.from_yaml(Path("config/default.yaml")).with_overrides(
        [
            "energy.basal_cost_per_node=0.0",
            "energy.feed_rate=0.0",
            "energy.scavenging_rate=0.0",
            "energy.photosynthesis_rate=0.0",
            "energy.predation_rate=2.0",
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

    event_types = [event.event_type for event in world.events]
    assert "predation_kill" in event_types


def test_world_logs_speciation_events_for_new_species_clusters() -> None:
    config = Config.from_yaml(Path("config/default.yaml")).with_overrides(
        [
            "energy.basal_cost_per_node=0.0",
            "energy.feed_rate=0.0",
            "energy.photosynthesis_rate=0.0",
        ]
    )
    node = NodeState(
        position=Vec2(10.0, 10.0),
        velocity=Vec2.zero(),
        accumulated_force=Vec2.zero(),
        drag_coeff=1.0,
        radius=1.0,
        node_type=NodeType.MOUTH,
    )
    creature = CreatureState(node_indices=(0,), energy=1.0)
    world = World(config=config, nodes=[node], creatures=[creature])
    world._known_species_ids = set()

    world.step()

    event_types = [event.event_type for event in world.events]
    assert "speciation" in event_types


def test_world_logs_species_extinction_events() -> None:
    config = Config.from_yaml(Path("config/default.yaml")).with_overrides(["creatures.min_population=0"])
    node = NodeState(
        position=Vec2(10.0, 10.0),
        velocity=Vec2.zero(),
        accumulated_force=Vec2.zero(),
        drag_coeff=1.0,
        radius=1.0,
    )
    creature = CreatureState(node_indices=(0,), energy=0.0001)
    world = World(config=config, nodes=[node], creatures=[creature])

    world.step()

    event_types = [event.event_type for event in world.events]
    assert "species_extinction" in event_types


def test_world_stats_capture_species_turnover_after_extinction() -> None:
    config = Config.from_yaml(Path("config/default.yaml")).with_overrides(["creatures.min_population=0"])
    node = NodeState(
        position=Vec2(10.0, 10.0),
        velocity=Vec2.zero(),
        accumulated_force=Vec2.zero(),
        drag_coeff=1.0,
        radius=1.0,
    )
    creature = CreatureState(node_indices=(0,), energy=0.0001)
    world = World(config=config, nodes=[node], creatures=[creature])

    world.step()

    stats = world.stats()
    assert stats.species_turnover == 1
    assert stats.observed_species_count == 1
    assert stats.peak_species_count == 1
    assert stats.peak_population == 1
    assert stats.population_variance > 0.0
    assert stats.population_capacity_fraction == 0.0
    assert stats.peak_population_capacity_fraction > 0.0
    assert stats.peak_species_fraction == 1.0
    assert stats.runaway_dominance_detected is False
    assert stats.mean_extinct_species_lifespan == 0.0


def test_world_detects_runaway_species_dominance() -> None:
    config = Config.from_yaml(Path("config/default.yaml")).with_overrides(
        [
            "energy.basal_cost_per_node=0.0",
            "energy.feed_rate=0.0",
            "energy.photosynthesis_rate=0.0",
        ]
    )
    node = NodeState(
        position=Vec2(10.0, 10.0),
        velocity=Vec2.zero(),
        accumulated_force=Vec2.zero(),
        drag_coeff=1.0,
        radius=1.0,
    )
    creature = CreatureState(node_indices=(0,), energy=1.0)
    world = World(config=config, nodes=[node], creatures=[creature])
    world._runaway_dominance_tick_threshold = 2

    world.step(3)

    stats = world.stats()
    assert stats.peak_species_fraction == 1.0
    assert stats.runaway_dominance_detected is True
