import random

from animalcula.sim.mutation import mutate_brain, mutate_edge, mutate_node
from animalcula.sim.types import BrainState, EdgeState, NodeState, Vec2


def test_mutate_node_changes_state_while_keeping_radius_positive() -> None:
    rng = random.Random(7)
    node = NodeState(
        position=Vec2(10.0, 10.0),
        velocity=Vec2.zero(),
        accumulated_force=Vec2.zero(),
        drag_coeff=1.0,
        radius=1.0,
    )

    mutated = mutate_node(
        node=node,
        rng=rng,
        position_sigma=0.5,
        radius_sigma=0.05,
    )

    assert mutated.position != node.position
    assert mutated.radius != node.radius
    assert mutated.radius > 0.0


def test_mutate_brain_changes_parameters_and_keeps_taus_positive() -> None:
    rng = random.Random(7)
    brain = BrainState(
        input_weights=((1.0, 2.0, 3.0),),
        recurrent_weights=((0.5,),),
        biases=(0.1,),
        time_constants=(1.0,),
        states=(0.0,),
        output_size=1,
    )

    mutated = mutate_brain(
        brain=brain,
        rng=rng,
        weight_sigma=0.1,
        bias_sigma=0.05,
        tau_sigma=0.02,
    )

    assert mutated.input_weights != brain.input_weights
    assert mutated.recurrent_weights != brain.recurrent_weights
    assert mutated.biases != brain.biases
    assert mutated.time_constants != brain.time_constants
    assert all(value > 0.0 for value in mutated.time_constants)


def test_mutate_edge_changes_motor_strength_while_keeping_it_non_negative() -> None:
    rng = random.Random(7)
    edge = EdgeState(a=0, b=1, rest_length=2.0, stiffness=1.0, has_motor=True, motor_strength=2.0)

    mutated = mutate_edge(edge=edge, rng=rng, motor_strength_sigma=0.2)

    assert mutated.motor_strength != edge.motor_strength
    assert mutated.motor_strength >= 0.0
