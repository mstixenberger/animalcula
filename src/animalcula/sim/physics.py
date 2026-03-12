"""Physics helpers for the overdamped simulation regime."""

from __future__ import annotations

from dataclasses import replace

from animalcula.sim.types import EdgeState, NodeState, Vec2


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


def apply_edge_springs(nodes: list[NodeState], edges: list[EdgeState]) -> list[NodeState]:
    updated_nodes = list(nodes)

    for edge in edges:
        force = spring_force(
            a=updated_nodes[edge.a].position,
            b=updated_nodes[edge.b].position,
            rest_length=edge.rest_length,
            stiffness=edge.stiffness,
        )
        updated_nodes[edge.a] = replace(
            updated_nodes[edge.a],
            accumulated_force=updated_nodes[edge.a].accumulated_force + force,
        )
        updated_nodes[edge.b] = replace(
            updated_nodes[edge.b],
            accumulated_force=updated_nodes[edge.b].accumulated_force - force,
        )

    return updated_nodes
