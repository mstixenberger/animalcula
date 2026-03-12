import random

from animalcula.sim.genome import CreatureGenome, decode_genome, encode_creature_genome, mutate_genome
from animalcula.sim.types import BrainState, CreatureState, EdgeState, NodeState, NodeType, Vec2


def test_encode_creature_genome_preserves_local_structure() -> None:
    nodes = [
        NodeState(
            position=Vec2(10.0, 10.0),
            velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(),
            drag_coeff=1.0,
            radius=1.0,
            node_type=NodeType.MOUTH,
        ),
        NodeState(
            position=Vec2(16.0, 10.0),
            velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(),
            drag_coeff=1.0,
            radius=1.5,
            node_type=NodeType.PHOTORECEPTOR,
        ),
    ]
    edges = [EdgeState(a=0, b=1, rest_length=6.0, stiffness=1.0, has_motor=True, motor_strength=2.0)]
    brain = BrainState(
        input_weights=((1.0, 2.0, 3.0, 4.0),),
        recurrent_weights=((0.5,),),
        biases=(0.1,),
        time_constants=(1.0,),
        states=(9.0,),
        output_size=1,
    )
    creature = CreatureState(node_indices=(0, 1), energy=5.0, brain=brain)

    genome = encode_creature_genome(nodes=nodes, edges=edges, creature=creature)

    assert isinstance(genome, CreatureGenome)
    assert len(genome.nodes) == 2
    assert genome.nodes[0].position == Vec2.zero()
    assert genome.nodes[1].position == Vec2(6.0, 0.0)
    assert genome.brain is not None
    assert genome.brain.states == (0.0,)


def test_decode_genome_rebuilds_nodes_edges_and_brain() -> None:
    genome = CreatureGenome(
        nodes=(
            CreatureGenome.NodeGene(position=Vec2.zero(), radius=1.0, node_type=NodeType.BODY),
            CreatureGenome.NodeGene(position=Vec2(3.0, 4.0), radius=1.5, node_type=NodeType.MOUTH),
        ),
        edges=(
            CreatureGenome.EdgeGene(
                a=0,
                b=1,
                rest_length=5.0,
                stiffness=2.0,
                has_motor=True,
                motor_strength=1.5,
            ),
        ),
        brain=CreatureGenome.BrainGene(
            input_weights=((1.0, 0.0),),
            recurrent_weights=((0.5,),),
            biases=(0.2,),
            time_constants=(1.0,),
            output_size=1,
        ),
    )

    nodes, edges, brain = decode_genome(genome=genome, anchor_position=Vec2(10.0, 20.0), drag_coeff=1.0)

    assert nodes[0].position == Vec2(10.0, 20.0)
    assert nodes[1].position == Vec2(13.0, 24.0)
    assert edges[0].rest_length == 5.0
    assert brain is not None
    assert brain.states == (0.0,)


def test_mutate_genome_changes_shape_or_brain_but_keeps_bounds_valid() -> None:
    rng = random.Random(7)
    genome = CreatureGenome(
        nodes=(CreatureGenome.NodeGene(position=Vec2.zero(), radius=1.0, node_type=NodeType.BODY),),
        edges=(),
        brain=CreatureGenome.BrainGene(
            input_weights=((1.0, 2.0),),
            recurrent_weights=((0.5,),),
            biases=(0.1,),
            time_constants=(1.0,),
            output_size=1,
        ),
    )

    mutated = mutate_genome(
        genome=genome,
        rng=rng,
        position_sigma=0.5,
        radius_sigma=0.05,
        weight_sigma=0.1,
        bias_sigma=0.05,
        tau_sigma=0.02,
        motor_strength_sigma=0.2,
    )

    assert mutated != genome
    assert mutated.nodes[0].radius > 0.0
    assert mutated.brain is not None
    assert all(value > 0.0 for value in mutated.brain.time_constants)
