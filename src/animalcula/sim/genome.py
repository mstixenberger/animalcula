"""Direct genome encoding for morphology and CTRNN parameters."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

from animalcula.sim.types import BrainState, CreatureState, EdgeState, NodeState, NodeType, Vec2


@dataclass(slots=True, frozen=True)
class GenomeNodeGene:
    position: Vec2
    radius: float
    node_type: NodeType


@dataclass(slots=True, frozen=True)
class GenomeEdgeGene:
    a: int
    b: int
    rest_length: float
    stiffness: float
    has_motor: bool = False
    motor_strength: float = 0.0


@dataclass(slots=True, frozen=True)
class GenomeBrainGene:
    input_weights: tuple[tuple[float, ...], ...]
    recurrent_weights: tuple[tuple[float, ...], ...]
    biases: tuple[float, ...]
    time_constants: tuple[float, ...]
    output_size: int
    states: tuple[float, ...] = ()


@dataclass(slots=True, frozen=True)
class CreatureGenome:
    nodes: tuple[GenomeNodeGene, ...]
    edges: tuple[GenomeEdgeGene, ...]
    brain: GenomeBrainGene | None = None


CreatureGenome.NodeGene = GenomeNodeGene  # type: ignore[attr-defined]
CreatureGenome.EdgeGene = GenomeEdgeGene  # type: ignore[attr-defined]
CreatureGenome.BrainGene = GenomeBrainGene  # type: ignore[attr-defined]


def encode_creature_genome(
    *,
    nodes: list[NodeState],
    edges: list[EdgeState],
    creature: CreatureState,
) -> CreatureGenome:
    if not creature.node_indices:
        return CreatureGenome(nodes=(), edges=(), brain=None)

    anchor = nodes[creature.node_indices[0]].position
    local_index = {node_index: local for local, node_index in enumerate(creature.node_indices)}
    genome_nodes = tuple(
        GenomeNodeGene(
            position=nodes[node_index].position - anchor,
            radius=nodes[node_index].radius,
            node_type=nodes[node_index].node_type,
        )
        for node_index in creature.node_indices
    )
    genome_edges = tuple(
        GenomeEdgeGene(
            a=local_index[edge.a],
            b=local_index[edge.b],
            rest_length=edge.rest_length,
            stiffness=edge.stiffness,
            has_motor=edge.has_motor,
            motor_strength=edge.motor_strength,
        )
        for edge in edges
        if edge.a in local_index and edge.b in local_index
    )
    genome_brain = None
    if creature.brain is not None:
        genome_brain = GenomeBrainGene(
            input_weights=creature.brain.input_weights,
            recurrent_weights=creature.brain.recurrent_weights,
            biases=creature.brain.biases,
            time_constants=creature.brain.time_constants,
            states=tuple(0.0 for _ in creature.brain.states),
            output_size=creature.brain.output_size,
        )
    return CreatureGenome(nodes=genome_nodes, edges=genome_edges, brain=genome_brain)


def decode_genome(
    *,
    genome: CreatureGenome,
    anchor_position: Vec2,
    drag_coeff: float,
) -> tuple[list[NodeState], list[EdgeState], BrainState | None]:
    nodes = [
        NodeState(
            position=anchor_position + node.position,
            velocity=Vec2.zero(),
            accumulated_force=Vec2.zero(),
            drag_coeff=drag_coeff,
            radius=node.radius,
            node_type=node.node_type,
        )
        for node in genome.nodes
    ]
    edges = [
        EdgeState(
            a=edge.a,
            b=edge.b,
            rest_length=edge.rest_length,
            stiffness=edge.stiffness,
            has_motor=edge.has_motor,
            motor_strength=edge.motor_strength,
        )
        for edge in genome.edges
    ]
    brain = None
    if genome.brain is not None:
        brain_states = (
            genome.brain.states
            if genome.brain.states
            else tuple(0.0 for _ in genome.brain.biases)
        )
        brain = BrainState(
            input_weights=genome.brain.input_weights,
            recurrent_weights=genome.brain.recurrent_weights,
            biases=genome.brain.biases,
            time_constants=genome.brain.time_constants,
            states=brain_states,
            output_size=genome.brain.output_size,
        )
    return nodes, edges, brain


def mutate_genome(
    *,
    genome: CreatureGenome,
    rng: random.Random,
    position_sigma: float,
    radius_sigma: float,
    weight_sigma: float,
    bias_sigma: float,
    tau_sigma: float,
    motor_strength_sigma: float,
) -> CreatureGenome:
    mutated_nodes = tuple(
        GenomeNodeGene(
            position=node.position
            + Vec2(
                rng.gauss(0.0, position_sigma),
                rng.gauss(0.0, position_sigma),
            ),
            radius=max(0.1, node.radius + rng.gauss(0.0, radius_sigma)),
            node_type=node.node_type,
        )
        for node in genome.nodes
    )
    mutated_edges = tuple(
        GenomeEdgeGene(
            a=edge.a,
            b=edge.b,
            rest_length=edge.rest_length,
            stiffness=edge.stiffness,
            has_motor=edge.has_motor,
            motor_strength=max(0.0, edge.motor_strength + rng.gauss(0.0, motor_strength_sigma))
            if edge.has_motor
            else edge.motor_strength,
        )
        for edge in genome.edges
    )
    mutated_brain = None
    if genome.brain is not None:
        mutated_brain = GenomeBrainGene(
            input_weights=tuple(
                tuple(weight + rng.gauss(0.0, weight_sigma) for weight in row)
                for row in genome.brain.input_weights
            ),
            recurrent_weights=tuple(
                tuple(weight + rng.gauss(0.0, weight_sigma) for weight in row)
                for row in genome.brain.recurrent_weights
            ),
            biases=tuple(bias + rng.gauss(0.0, bias_sigma) for bias in genome.brain.biases),
            time_constants=tuple(
                max(0.1, tau + rng.gauss(0.0, tau_sigma)) for tau in genome.brain.time_constants
            ),
            states=genome.brain.states,
            output_size=genome.brain.output_size,
        )
    return CreatureGenome(nodes=mutated_nodes, edges=mutated_edges, brain=mutated_brain)


def genome_to_dict(genome: CreatureGenome | None) -> dict[str, Any] | None:
    if genome is None:
        return None
    return {
        "nodes": [
            {
                "position": [node.position.x, node.position.y],
                "radius": node.radius,
                "node_type": node.node_type.value,
            }
            for node in genome.nodes
        ],
        "edges": [
            {
                "a": edge.a,
                "b": edge.b,
                "rest_length": edge.rest_length,
                "stiffness": edge.stiffness,
                "has_motor": edge.has_motor,
                "motor_strength": edge.motor_strength,
            }
            for edge in genome.edges
        ],
        "brain": None
        if genome.brain is None
        else {
            "input_weights": [list(row) for row in genome.brain.input_weights],
            "recurrent_weights": [list(row) for row in genome.brain.recurrent_weights],
            "biases": list(genome.brain.biases),
            "time_constants": list(genome.brain.time_constants),
            "states": list(genome.brain.states),
            "output_size": genome.brain.output_size,
        },
    }


def genome_from_dict(payload: dict[str, Any] | None) -> CreatureGenome | None:
    if payload is None:
        return None
    return CreatureGenome(
        nodes=tuple(
            GenomeNodeGene(
                position=Vec2(*node["position"]),
                radius=node["radius"],
                node_type=NodeType(node["node_type"]),
            )
            for node in payload["nodes"]
        ),
        edges=tuple(
            GenomeEdgeGene(
                a=edge["a"],
                b=edge["b"],
                rest_length=edge["rest_length"],
                stiffness=edge["stiffness"],
                has_motor=edge.get("has_motor", False),
                motor_strength=edge.get("motor_strength", 0.0),
            )
            for edge in payload["edges"]
        ),
        brain=None
        if payload.get("brain") is None
        else GenomeBrainGene(
            input_weights=tuple(tuple(row) for row in payload["brain"]["input_weights"]),
            recurrent_weights=tuple(tuple(row) for row in payload["brain"]["recurrent_weights"]),
            biases=tuple(payload["brain"]["biases"]),
            time_constants=tuple(payload["brain"]["time_constants"]),
            states=tuple(payload["brain"].get("states", [])),
            output_size=payload["brain"]["output_size"],
        ),
    )
