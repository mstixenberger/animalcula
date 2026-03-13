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
    source_a = nutrient_grid.position_for_cell(*nutrient_source_cells[0])
    bright_anchor = Vec2(world_width - 15.0, world_height * 0.5)

    nodes: list[NodeState] = []
    edges: list[EdgeState] = []
    creatures: list[CreatureState] = []

    # Alga: tripod light harvester near the bright edge.
    alga_nodes = [
        _node(bright_anchor, node_type=NodeType.PHOTORECEPTOR),
        _node(Vec2(bright_anchor.x - 5.0, bright_anchor.y + 3.0)),
        _node(Vec2(bright_anchor.x - 5.0, bright_anchor.y - 3.0)),
    ]
    _append_creature(
        nodes,
        edges,
        creatures,
        alga_nodes,
        [(0, 1), (0, 2), (1, 2)],
        energy=2.5,
        color_rgb=(214, 180, 70),
    )

    # Worm: four-segment grazer with a visible head/tail body plan.
    grazer_nodes = [
        _node(source_a, node_type=NodeType.MOUTH),
        _node(Vec2(source_a.x + 4.0, source_a.y + 1.5)),
        _node(Vec2(source_a.x + 8.0, source_a.y - 1.5)),
        _node(Vec2(source_a.x + 12.0, source_a.y + 0.5), node_type=NodeType.SENSOR),
    ]
    _append_creature(
        nodes,
        edges,
        creatures,
        grazer_nodes,
        [(0, 1), (1, 2), (2, 3)],
        energy=2.0,
        brain=_worm_brain(light_gain=0.2, nutrient_gain=2.8),
        color_rgb=(54, 162, 147),
    )

    # Amoeba: triangular predator seeded on the grazer basin so predation can actually bootstrap.
    amoeba_nodes = [
        _node(source_a, node_type=NodeType.MOUTH),
        _node(Vec2(source_a.x + 5.0, source_a.y + 4.5), node_type=NodeType.GRIPPER),
        _node(Vec2(source_a.x - 5.0, source_a.y + 4.5)),
    ]
    _append_creature(
        nodes,
        edges,
        creatures,
        amoeba_nodes,
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
