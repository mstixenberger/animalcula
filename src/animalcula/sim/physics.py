"""Physics helpers for the overdamped simulation regime."""

from __future__ import annotations

from dataclasses import replace

from animalcula.sim.types import NodeState, Vec2


def spring_force(a: Vec2, b: Vec2, rest_length: float, stiffness: float) -> Vec2:
    displacement = b - a
    distance = displacement.magnitude()
    if distance == 0.0:
        return Vec2.zero()

    extension = distance - rest_length
    if extension == 0.0:
        return Vec2.zero()

    direction = displacement.normalized()
    return direction * (stiffness * extension)


def apply_overdamped_dynamics(node: NodeState, dt: float) -> NodeState:
    if node.drag_coeff <= 0.0:
        raise ValueError("drag_coeff must be positive")

    velocity = node.accumulated_force / node.drag_coeff
    position = node.position + (velocity * dt)
    return replace(
        node,
        position=position,
        velocity=velocity,
        accumulated_force=Vec2.zero(),
    )
