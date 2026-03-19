import math

import pytest

from animalcula.sim.physics import (
    apply_edge_springs,
    apply_node_repulsion,
    apply_motor_forces,
    apply_overdamped_dynamics,
    creature_heading,
    spring_force,
)
from animalcula.sim.types import CreatureState, EdgeState, NodeState, Vec2


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


def test_overdamped_dynamics_applies_drag_multiplier() -> None:
    node = NodeState(
        position=Vec2.zero(),
        velocity=Vec2.zero(),
        accumulated_force=Vec2(4.0, 0.0),
        drag_coeff=2.0,
        radius=1.0,
    )

    updated = apply_overdamped_dynamics(node=node, dt=0.5, drag_multiplier=2.0)

    assert updated.velocity == Vec2(1.0, 0.0)
    assert updated.position == Vec2(0.5, 0.0)


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


def _com(nodes: list[NodeState]) -> Vec2:
    """Compute centre-of-mass (unweighted) of a node list."""
    x = sum(n.position.x for n in nodes) / len(nodes)
    y = sum(n.position.y for n in nodes) / len(nodes)
    return Vec2(x, y)


def _run_scallop_cycle(
    drag_a: float,
    drag_b: float,
    *,
    rest_length: float = 5.0,
    stiffness: float = 10.0,
    motor_strength: float = 5.0,
    dt: float = 0.01,
    cycle_ticks: int = 2000,
) -> float:
    """Simulate a 2-node, 1-motor creature through a full sinusoidal motor cycle.

    Returns the x-component of the COM displacement after one full cycle.
    """
    nodes = [
        NodeState(
            position=Vec2(0.0, 0.0),
            velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(),
            drag_coeff=drag_a,
            radius=0.5,
        ),
        NodeState(
            position=Vec2(rest_length, 0.0),
            velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(),
            drag_coeff=drag_b,
            radius=0.5,
        ),
    ]
    edges = [
        EdgeState(
            a=0,
            b=1,
            rest_length=rest_length,
            stiffness=stiffness,
            has_motor=True,
            motor_strength=motor_strength,
        )
    ]

    com_start = _com(nodes)

    for tick in range(cycle_ticks):
        phase = 2.0 * math.pi * tick / cycle_ticks
        motor_output = math.sin(phase)

        nodes = apply_edge_springs(nodes=nodes, edges=edges)
        nodes = apply_motor_forces(nodes=nodes, edges=edges, edge_outputs={0: motor_output})
        nodes = [apply_overdamped_dynamics(node=n, dt=dt) for n in nodes]

    com_end = _com(nodes)
    return com_end.x - com_start.x


def test_scallop_theorem_symmetric_drag_no_net_displacement() -> None:
    """A 2-node swimmer with equal drag and symmetric oscillation obeys the
    scallop theorem: zero net displacement after a full cycle."""
    displacement = _run_scallop_cycle(drag_a=1.0, drag_b=1.0)
    assert abs(displacement) < 0.01, f"expected ~0 displacement, got {displacement}"


def test_scallop_theorem_asymmetric_drag_still_zero_for_one_dof() -> None:
    """Even with unequal drag a 1-DOF (2-node) reciprocal swimmer cannot
    achieve net displacement — the scallop theorem is about shape-space
    dimensionality, not drag symmetry."""
    displacement = _run_scallop_cycle(drag_a=1.0, drag_b=3.0)
    assert abs(displacement) < 0.01, f"expected ~0 displacement, got {displacement}"


def test_multi_dof_asymmetric_drag_enables_locomotion() -> None:
    """A 3-node, 2-motor swimmer with phase-offset oscillation AND unequal
    drag produces net displacement.  In our overdamped model (purely internal
    forces) locomotion requires both multiple DOF and drag asymmetry:
    multiple DOF creates time-irreversible shape sequences, and drag
    asymmetry breaks the force-cancellation that makes COM stationary
    when all drags are equal."""
    rest_length = 5.0
    stiffness = 10.0
    motor_strength = 5.0
    dt = 0.01
    cycle_ticks = 2000

    nodes = [
        NodeState(
            position=Vec2(0.0, 0.0),
            velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(),
            drag_coeff=1.0,
            radius=0.5,
        ),
        NodeState(
            position=Vec2(rest_length, 0.0),
            velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(),
            drag_coeff=3.0,
            radius=0.5,
        ),
        NodeState(
            position=Vec2(2.0 * rest_length, 0.0),
            velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(),
            drag_coeff=1.0,
            radius=0.5,
        ),
    ]
    edges = [
        EdgeState(
            a=0, b=1, rest_length=rest_length, stiffness=stiffness,
            has_motor=True, motor_strength=motor_strength,
        ),
        EdgeState(
            a=1, b=2, rest_length=rest_length, stiffness=stiffness,
            has_motor=True, motor_strength=motor_strength,
        ),
    ]

    com_start = _com(nodes)

    for tick in range(cycle_ticks):
        phase = 2.0 * math.pi * tick / cycle_ticks
        motor_output_0 = math.sin(phase)
        motor_output_1 = math.sin(phase + math.pi / 2.0)  # 90° phase offset

        nodes = apply_edge_springs(nodes=nodes, edges=edges)
        nodes = apply_motor_forces(
            nodes=nodes, edges=edges,
            edge_outputs={0: motor_output_0, 1: motor_output_1},
        )
        nodes = [apply_overdamped_dynamics(node=n, dt=dt) for n in nodes]

    com_end = _com(nodes)
    displacement = com_end.x - com_start.x
    assert abs(displacement) > 0.01, (
        f"expected nonzero displacement from multi-DOF + asymmetric drag, got {displacement}"
    )


def _make_node(x: float, y: float) -> NodeState:
    return NodeState(
        position=Vec2(x, y),
        velocity=Vec2.zero(),
        accumulated_force=Vec2.zero(),
        drag_coeff=1.0,
        radius=1.0,
    )


def test_creature_heading_points_from_com_toward_node0() -> None:
    """Heading should be the unit vector from COM to node 0."""
    nodes = [_make_node(10.0, 0.0), _make_node(0.0, 0.0)]
    creature = CreatureState(node_indices=(0, 1), energy=1.0)
    heading = creature_heading(nodes, creature)
    assert math.isclose(heading.x, 1.0, abs_tol=1e-9)
    assert math.isclose(heading.y, 0.0, abs_tol=1e-9)


def test_creature_heading_is_unit_vector() -> None:
    nodes = [_make_node(3.0, 4.0), _make_node(0.0, 0.0)]
    creature = CreatureState(node_indices=(0, 1), energy=1.0)
    heading = creature_heading(nodes, creature)
    assert math.isclose(heading.magnitude(), 1.0, rel_tol=1e-9)


def test_creature_heading_fallback_when_node0_at_com() -> None:
    """When node 0 is exactly at the COM, heading should fall back to (1, 0)."""
    nodes = [_make_node(5.0, 5.0)]
    creature = CreatureState(node_indices=(0,), energy=1.0)
    heading = creature_heading(nodes, creature)
    assert heading == Vec2(1.0, 0.0)
