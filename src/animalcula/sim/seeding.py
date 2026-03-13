"""Deterministic starter population builders."""

from __future__ import annotations

from animalcula.sim.fields import Grid2D
from animalcula.sim.types import BrainState, CreatureState, EdgeState, NodeState, NodeType, Vec2


def build_demo_archetypes(
    world_width: float,
    world_height: float,
    nutrient_grid: Grid2D,
    nutrient_source_cells: list[tuple[int, int]],
) -> tuple[list[NodeState], list[EdgeState], list[CreatureState]]:
    nutrient_anchors = [
        nutrient_grid.position_for_cell(*cell)
        for cell in nutrient_source_cells[:3]
    ]
    while len(nutrient_anchors) < 3:
        base = nutrient_anchors[0] if nutrient_anchors else Vec2(world_width * 0.3, world_height * 0.5)
        nutrient_anchors.append(Vec2(base.x + (12.0 * len(nutrient_anchors)), base.y + (8.0 * len(nutrient_anchors))))
    bright_anchors = [
        Vec2(world_width - 15.0, world_height * 0.35),
        Vec2(world_width - 15.0, world_height * 0.5),
        Vec2(world_width - 15.0, world_height * 0.65),
    ]

    nodes: list[NodeState] = []
    edges: list[EdgeState] = []
    creatures: list[CreatureState] = []

    alga_template = [
        _node(Vec2(0.0, 0.0), node_type=NodeType.PHOTORECEPTOR),
        _node(Vec2(-5.0, 3.0)),
        _node(Vec2(-5.0, -3.0)),
    ]
    worm_template = [
        _node(Vec2(0.0, 0.0), node_type=NodeType.MOUTH),
        _node(Vec2(4.0, 1.5)),
        _node(Vec2(8.0, -1.5)),
        _node(Vec2(12.0, 0.5), node_type=NodeType.SENSOR),
    ]
    amoeba_template = [
        _node(Vec2(0.0, 0.0), node_type=NodeType.MOUTH),
        _node(Vec2(5.0, 4.5), node_type=NodeType.GRIPPER),
        _node(Vec2(-5.0, 4.5)),
    ]

    for bright_anchor in bright_anchors:
        _append_creature(
            nodes,
            edges,
            creatures,
            _translated_nodes(alga_template, bright_anchor),
            [(0, 1), (0, 2), (1, 2)],
            energy=2.5,
            color_rgb=(214, 180, 70),
        )

    for source_anchor in nutrient_anchors:
        _append_creature(
            nodes,
            edges,
            creatures,
            _translated_nodes(worm_template, source_anchor),
            [(0, 1), (1, 2), (2, 3)],
            energy=2.0,
            brain=_worm_brain(light_gain=0.2, nutrient_gain=2.8),
            color_rgb=(54, 162, 147),
        )
        _append_creature(
            nodes,
            edges,
            creatures,
            _translated_nodes(amoeba_template, source_anchor),
            [(0, 1), (1, 2), (2, 0)],
            energy=2.0,
            brain=_predator_brain(light_gain=1.0, nutrient_gain=1.0),
            color_rgb=(198, 73, 92),
        )

    return nodes, edges, creatures


def _node(
    position: Vec2,
    *,
    node_type: NodeType = NodeType.BODY,
    radius: float = 1.4,
) -> NodeState:
    return NodeState(
        position=position,
        velocity=Vec2.zero(),
        accumulated_force=Vec2.zero(),
        drag_coeff=1.0,
        radius=radius,
        node_type=node_type,
    )


def _append_creature(
    world_nodes: list[NodeState],
    world_edges: list[EdgeState],
    world_creatures: list[CreatureState],
    creature_nodes: list[NodeState],
    local_edges: list[tuple[int, int]],
    energy: float,
    brain: BrainState | None = None,
    color_rgb: tuple[int, int, int] = (160, 175, 190),
) -> None:
    node_offset = len(world_nodes)
    world_nodes.extend(creature_nodes)
    world_edges.extend(
        EdgeState(
            a=node_offset + a,
            b=node_offset + b,
            rest_length=(world_nodes[node_offset + b].position - world_nodes[node_offset + a].position).magnitude(),
            stiffness=1.0,
            has_motor=brain is not None,
            motor_strength=2.0 if brain is not None else 0.0,
        )
        for a, b in local_edges
    )
    world_creatures.append(
        CreatureState(
            node_indices=tuple(range(node_offset, node_offset + len(creature_nodes))),
            energy=energy,
            brain=brain,
            color_rgb=color_rgb,
        )
    )


def _translated_nodes(template: list[NodeState], anchor: Vec2) -> list[NodeState]:
    return [
        _node(
            Vec2(anchor.x + node.position.x, anchor.y + node.position.y),
            node_type=node.node_type,
            radius=node.radius,
        )
        for node in template
    ]


def _worm_brain(light_gain: float, nutrient_gain: float) -> BrainState:
    return BrainState(
        input_weights=(
            (light_gain, nutrient_gain, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (light_gain, nutrient_gain, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (light_gain * 0.5, nutrient_gain, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, nutrient_gain * 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ),
        recurrent_weights=(
            (1.25, -0.9, 0.45, 0.0),
            (0.55, 1.2, -0.9, 0.0),
            (0.0, 0.55, 1.2, -0.7),
            (0.0, 0.0, 0.4, 1.0),
        ),
        biases=(0.8, -0.6, 0.5, 1.6),
        time_constants=(0.8, 1.2, 1.0, 1.0),
        states=(1.5, -1.0, 1.0, 2.0),
        output_size=4,
    )


def _predator_brain(light_gain: float, nutrient_gain: float) -> BrainState:
    return BrainState(
        input_weights=(
            (light_gain, nutrient_gain, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (light_gain, nutrient_gain, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (light_gain, nutrient_gain, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, nutrient_gain, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, nutrient_gain, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ),
        recurrent_weights=(
            (1.0, 0.2, 0.0, 0.0, 0.0, 0.0),
            (0.2, 1.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.2, 1.0, 0.0, 0.0),
            (0.0, 0.0, 0.4, 0.2, 1.0, 0.0),
            (0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ),
        biases=(0.0, 0.0, 0.0, 0.5, 1.0, -2.0),
        time_constants=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        states=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        output_size=6,
    )
