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
    source_b = nutrient_grid.position_for_cell(*nutrient_source_cells[1])
    bright_anchor = Vec2(world_width - 15.0, world_height * 0.5)

    nodes: list[NodeState] = []
    edges: list[EdgeState] = []
    creatures: list[CreatureState] = []

    # Alga: passive light harvester near the bright edge.
    alga_nodes = [
        NodeState(
            position=bright_anchor,
            velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(),
            drag_coeff=1.0,
            radius=1.0,
            node_type=NodeType.PHOTORECEPTOR,
        ),
        NodeState(
            position=Vec2(bright_anchor.x - 6.0, bright_anchor.y),
            velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(),
            drag_coeff=1.0,
            radius=1.0,
        ),
    ]
    _append_creature(nodes, edges, creatures, alga_nodes, [(0, 1)], energy=1.0)

    # Grazer: mouth anchored on a nutrient source.
    grazer_nodes = [
        NodeState(
            position=source_a,
            velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(),
            drag_coeff=1.0,
            radius=1.0,
            node_type=NodeType.MOUTH,
        ),
        NodeState(
            position=Vec2(source_a.x + 5.0, source_a.y),
            velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(),
            drag_coeff=1.0,
            radius=1.0,
        ),
    ]
    _append_creature(
        nodes,
        edges,
        creatures,
        grazer_nodes,
        [(0, 1)],
        energy=1.0,
        brain=_simple_motor_brain(light_gain=0.0, nutrient_gain=2.0),
    )

    # Amoeba-lite: mixed feeder with both nutrient and light access.
    amoeba_nodes = [
        NodeState(
            position=source_b,
            velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(),
            drag_coeff=1.0,
            radius=1.0,
            node_type=NodeType.MOUTH,
        ),
        NodeState(
            position=Vec2(source_b.x + 4.0, source_b.y + 4.0),
            velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(),
            drag_coeff=1.0,
            radius=1.0,
            node_type=NodeType.PHOTORECEPTOR,
        ),
        NodeState(
            position=Vec2(source_b.x - 4.0, source_b.y + 4.0),
            velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(),
            drag_coeff=1.0,
            radius=1.0,
        ),
    ]
    _append_creature(
        nodes,
        edges,
        creatures,
        amoeba_nodes,
        [(0, 1), (0, 2)],
        energy=1.0,
        brain=_simple_motor_brain(light_gain=1.0, nutrient_gain=1.0),
    )

    return nodes, edges, creatures


def _append_creature(
    world_nodes: list[NodeState],
    world_edges: list[EdgeState],
    world_creatures: list[CreatureState],
    creature_nodes: list[NodeState],
    local_edges: list[tuple[int, int]],
    energy: float,
    brain: BrainState | None = None,
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
        )
    )


def _simple_motor_brain(light_gain: float, nutrient_gain: float) -> BrainState:
    return BrainState(
        input_weights=((light_gain, nutrient_gain, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),),
        recurrent_weights=((1.0,),),
        biases=(0.0,),
        time_constants=(1.0,),
        states=(0.0,),
        output_size=1,
    )
