"""Physics helpers for the overdamped simulation regime."""

from __future__ import annotations

from dataclasses import replace

from animalcula.sim.types import CreatureState, EdgeState, GripLatch, NodeState, Vec2


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


def apply_overdamped_dynamics(node: NodeState, dt: float, drag_multiplier: float = 1.0) -> NodeState:
    if node.drag_coeff <= 0.0:
        raise ValueError("drag_coeff must be positive")
    if drag_multiplier <= 0.0:
        raise ValueError("drag_multiplier must be positive")

    velocity = node.accumulated_force / (node.drag_coeff * drag_multiplier)
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


def apply_motor_forces(
    nodes: list[NodeState],
    edges: list[EdgeState],
    edge_outputs: dict[int, float],
) -> list[NodeState]:
    updated_nodes = list(nodes)

    for edge_index, output in edge_outputs.items():
        edge = edges[edge_index]
        if not edge.has_motor:
            continue
        direction = (updated_nodes[edge.b].position - updated_nodes[edge.a].position).normalized()
        force = direction * (edge.motor_strength * output)
        updated_nodes[edge.a] = replace(
            updated_nodes[edge.a],
            accumulated_force=updated_nodes[edge.a].accumulated_force + force,
        )
        updated_nodes[edge.b] = replace(
            updated_nodes[edge.b],
            accumulated_force=updated_nodes[edge.b].accumulated_force - force,
        )

    return updated_nodes


def apply_node_repulsion(nodes: list[NodeState], strength: float) -> list[NodeState]:
    if strength <= 0.0:
        return list(nodes)

    updated_nodes = list(nodes)

    for a_index in range(len(updated_nodes)):
        for b_index in range(a_index + 1, len(updated_nodes)):
            displacement = updated_nodes[b_index].position - updated_nodes[a_index].position
            distance = displacement.magnitude()
            minimum_distance = updated_nodes[a_index].radius + updated_nodes[b_index].radius
            overlap = minimum_distance - distance
            if overlap <= 0.0:
                continue

            direction = displacement.normalized() if distance > 0.0 else Vec2(1.0, 0.0)
            force = direction * (strength * overlap)
            updated_nodes[a_index] = replace(
                updated_nodes[a_index],
                accumulated_force=updated_nodes[a_index].accumulated_force - force,
            )
            updated_nodes[b_index] = replace(
                updated_nodes[b_index],
                accumulated_force=updated_nodes[b_index].accumulated_force + force,
            )

    return updated_nodes


def apply_grip_latches(
    nodes: list[NodeState],
    latches: list[GripLatch],
    stiffness: float,
) -> list[NodeState]:
    if stiffness <= 0.0 or not latches:
        return list(nodes)

    updated_nodes = list(nodes)

    for latch in latches:
        force = spring_force(
            a=updated_nodes[latch.node_a_index].position,
            b=updated_nodes[latch.node_b_index].position,
            rest_length=latch.rest_length,
            stiffness=stiffness,
        )
        updated_nodes[latch.node_a_index] = replace(
            updated_nodes[latch.node_a_index],
            accumulated_force=updated_nodes[latch.node_a_index].accumulated_force + force,
        )
        updated_nodes[latch.node_b_index] = replace(
            updated_nodes[latch.node_b_index],
            accumulated_force=updated_nodes[latch.node_b_index].accumulated_force - force,
        )

    return updated_nodes


def apply_wall_repulsion(
    nodes: list[NodeState],
    width: float,
    height: float,
    strength: float,
    margin: float,
) -> list[NodeState]:
    """Apply soft 1/r² repulsive force from world boundaries."""
    if strength <= 0.0 or margin <= 0.0:
        return list(nodes)

    epsilon = 0.01
    max_force = strength * 100.0  # cap to prevent explosion
    updated_nodes = list(nodes)

    for i, node in enumerate(updated_nodes):
        fx, fy = 0.0, 0.0
        # Left wall (x=0)
        dist = node.position.x
        if 0.0 < dist < margin:
            fx += min(strength / max(dist * dist, epsilon), max_force)
        elif dist <= 0.0:
            fx += max_force
        # Right wall (x=width)
        dist = width - node.position.x
        if 0.0 < dist < margin:
            fx -= min(strength / max(dist * dist, epsilon), max_force)
        elif dist <= 0.0:
            fx -= max_force
        # Bottom wall (y=0)
        dist = node.position.y
        if 0.0 < dist < margin:
            fy += min(strength / max(dist * dist, epsilon), max_force)
        elif dist <= 0.0:
            fy += max_force
        # Top wall (y=height)
        dist = height - node.position.y
        if 0.0 < dist < margin:
            fy -= min(strength / max(dist * dist, epsilon), max_force)
        elif dist <= 0.0:
            fy -= max_force

        if fx != 0.0 or fy != 0.0:
            updated_nodes[i] = replace(
                node,
                accumulated_force=node.accumulated_force + Vec2(fx, fy),
            )

    return updated_nodes


def apply_obstacle_repulsion(
    nodes: list[NodeState],
    obstacles: list,
    strength: float,
) -> list[NodeState]:
    """Apply soft repulsion from static obstacles (same overlap pattern as node-node contact)."""
    if strength <= 0.0 or not obstacles:
        return list(nodes)

    updated_nodes = list(nodes)

    for i, node in enumerate(updated_nodes):
        for obstacle in obstacles:
            displacement = node.position - Vec2(obstacle.x, obstacle.y)
            distance = displacement.magnitude()
            minimum_distance = node.radius + obstacle.radius
            overlap = minimum_distance - distance
            if overlap <= 0.0:
                continue
            direction = displacement.normalized() if distance > 0.0 else Vec2(1.0, 0.0)
            force = direction * (strength * overlap)
            updated_nodes[i] = replace(
                updated_nodes[i],
                accumulated_force=updated_nodes[i].accumulated_force + force,
            )

    return updated_nodes


def creature_heading(nodes: list[NodeState], creature: CreatureState) -> Vec2:
    """Return the heading unit vector for a creature: (node0_pos - COM).normalized().

    Falls back to Vec2(1, 0) when the creature has a single node or node 0
    coincides with the COM.
    """
    if not creature.node_indices:
        return Vec2(1.0, 0.0)

    com_x = sum(nodes[i].position.x for i in creature.node_indices) / len(creature.node_indices)
    com_y = sum(nodes[i].position.y for i in creature.node_indices) / len(creature.node_indices)
    node0 = nodes[creature.node_indices[0]].position
    direction = Vec2(node0.x - com_x, node0.y - com_y)
    normalized = direction.normalized()
    if normalized == Vec2.zero():
        return Vec2(1.0, 0.0)
    return normalized
