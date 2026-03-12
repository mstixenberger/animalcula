import random

from animalcula.sim.mutation import mutate_node
from animalcula.sim.types import NodeState, Vec2


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
