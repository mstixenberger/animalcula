import math

import pytest

from animalcula.sim.physics import (
    apply_edge_springs,
    apply_node_repulsion,
    apply_motor_forces,
    apply_overdamped_dynamics,
    spring_force,
)
from animalcula.sim.types import EdgeState, NodeState, Vec2


def test_spring_force_is_zero_at_rest_length() -> None:
    force = spring_force(
        a=Vec2(0.0, 0.0),
        b=Vec2(2.0, 0.0),
        rest_length=2.0,
        stiffness=3.0,
    )

    assert force == Vec2.zero()


def test_spring_force_pulls_stretched_nodes_together() -> None:
    force = spring_force(
        a=Vec2(0.0, 0.0),
        b=Vec2(3.0, 4.0),
        rest_length=2.0,
        stiffness=2.0,
    )

    assert force.x > 0.0
    assert force.y > 0.0
    assert math.isclose(force.magnitude(), 6.0, rel_tol=1e-6)


def test_overdamped_dynamics_updates_velocity_and_position() -> None:
    node = NodeState(
        position=Vec2(1.0, -1.0),
        velocity=Vec2.zero(),
        accumulated_force=Vec2(4.0, -2.0),
        drag_coeff=2.0,
        radius=1.0,
    )

    updated = apply_overdamped_dynamics(node=node, dt=0.5)

    assert updated.velocity == Vec2(2.0, -1.0)
    assert updated.position == Vec2(2.0, -1.5)
    assert updated.accumulated_force == Vec2.zero()


def test_overdamped_dynamics_requires_positive_drag() -> None:
    node = NodeState(
        position=Vec2.zero(),
        velocity=Vec2.zero(),
        accumulated_force=Vec2(1.0, 0.0),
        drag_coeff=0.0,
        radius=1.0,
    )

    with pytest.raises(ValueError, match="drag_coeff must be positive"):
        apply_overdamped_dynamics(node=node, dt=0.1)


def test_apply_edge_springs_accumulates_equal_and_opposite_forces() -> None:
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

    updated = apply_edge_springs(nodes=nodes, edges=edges)

    assert updated[0].accumulated_force == Vec2(4.0, 0.0)
    assert updated[1].accumulated_force == Vec2(-4.0, 0.0)


def test_apply_motor_forces_pushes_nodes_along_motorized_edge() -> None:
    nodes = [
        NodeState(
            position=Vec2(0.0, 0.0),
            velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(),
            drag_coeff=1.0,
            radius=1.0,
        ),
        NodeState(
            position=Vec2(2.0, 0.0),
            velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(),
            drag_coeff=1.0,
            radius=1.0,
        ),
    ]
    edges = [
        EdgeState(a=0, b=1, rest_length=2.0, stiffness=1.0, has_motor=True, motor_strength=3.0)
    ]

    updated = apply_motor_forces(nodes=nodes, edges=edges, edge_outputs={0: 1.0})

    assert updated[0].accumulated_force == Vec2(3.0, 0.0)
    assert updated[1].accumulated_force == Vec2(-3.0, 0.0)


def test_apply_node_repulsion_pushes_overlapping_nodes_apart() -> None:
    nodes = [
        NodeState(
            position=Vec2(0.0, 0.0),
            velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(),
            drag_coeff=1.0,
            radius=1.0,
        ),
        NodeState(
            position=Vec2(1.0, 0.0),
            velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(),
            drag_coeff=1.0,
            radius=1.0,
        ),
    ]

    updated = apply_node_repulsion(nodes=nodes, strength=2.0)

    assert updated[0].accumulated_force.x < 0.0
    assert updated[1].accumulated_force.x > 0.0
