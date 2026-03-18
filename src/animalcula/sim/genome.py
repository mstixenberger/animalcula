"""Direct genome encoding for morphology and CTRNN parameters."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import random
from collections import defaultdict
from typing import Any

from animalcula.sim.types import (
    DEFAULT_LINEAGE_COLOR_RGB,
    BrainState,
    CreatureState,
    EdgeState,
    NodeState,
    NodeType,
    Vec2,
)


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
class GenomeVisualGene:
    silhouette_scale: float = 1.0
    glyph_scale: float = 1.0
    band_count: int = 2
    band_offset: float = 0.0


@dataclass(slots=True, frozen=True)
class CreatureGenome:
    nodes: tuple[GenomeNodeGene, ...]
    edges: tuple[GenomeEdgeGene, ...]
    brain: GenomeBrainGene | None = None
    color_rgb: tuple[int, int, int] = DEFAULT_LINEAGE_COLOR_RGB
    visuals: GenomeVisualGene = GenomeVisualGene()


CreatureGenome.NodeGene = GenomeNodeGene  # type: ignore[attr-defined]
CreatureGenome.EdgeGene = GenomeEdgeGene  # type: ignore[attr-defined]
CreatureGenome.BrainGene = GenomeBrainGene  # type: ignore[attr-defined]
CreatureGenome.VisualGene = GenomeVisualGene  # type: ignore[attr-defined]

MUTABLE_NODE_TYPES: tuple[NodeType, ...] = (
    NodeType.BODY,
    NodeType.MOUTH,
    NodeType.GRIPPER,
    NodeType.SENSOR,
    NodeType.PHOTORECEPTOR,
)


def required_control_outputs(genome: CreatureGenome) -> int:
    return (
        sum(1 for edge in genome.edges if edge.has_motor)
        + sum(1 for node in genome.nodes if node.node_type == NodeType.GRIPPER)
        + sum(1 for node in genome.nodes if node.node_type == NodeType.MOUTH)
    )


def _mutated_node_type(current: NodeType, rng: random.Random) -> NodeType:
    choices = [node_type for node_type in MUTABLE_NODE_TYPES if node_type != current]
    return rng.choice(choices)


def _clamp_rgb_channel(value: float) -> int:
    return max(0, min(255, int(round(value))))


def _mutate_color_rgb(
    color_rgb: tuple[int, int, int],
    *,
    rng: random.Random,
    sigma: float,
) -> tuple[int, int, int]:
    if sigma <= 0.0:
        return color_rgb
    return tuple(_clamp_rgb_channel(channel + rng.gauss(0.0, sigma)) for channel in color_rgb)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _mutate_visuals(visuals: GenomeVisualGene, *, rng: random.Random) -> GenomeVisualGene:
    band_count = visuals.band_count
    if rng.random() < 0.25:
        band_count += 1 if rng.random() < 0.5 else -1
    return GenomeVisualGene(
        silhouette_scale=_clamp(visuals.silhouette_scale + rng.gauss(0.0, 0.08), 0.85, 1.5),
        glyph_scale=_clamp(visuals.glyph_scale + rng.gauss(0.0, 0.1), 0.85, 1.75),
        band_count=max(1, min(4, band_count)),
        band_offset=(visuals.band_offset + rng.gauss(0.0, 0.12)) % 1.0,
    )


def _resize_brain_for_outputs(
    brain: GenomeBrainGene,
    target_output_size: int,
) -> GenomeBrainGene:
    current_hidden = len(brain.biases)
    target_hidden = max(current_hidden, target_output_size)
    input_size = len(brain.input_weights[0]) if brain.input_weights else 0

    input_weights = [list(row) for row in brain.input_weights]
    recurrent_weights = [list(row) for row in brain.recurrent_weights]

    for row in input_weights:
        if len(row) < input_size:
            row.extend(0.0 for _ in range(input_size - len(row)))
    while len(input_weights) < target_hidden:
        input_weights.append([0.0] * input_size)

    for row in recurrent_weights:
        if len(row) < target_hidden:
            row.extend(0.0 for _ in range(target_hidden - len(row)))
    while len(recurrent_weights) < target_hidden:
        recurrent_weights.append([0.0] * target_hidden)

    biases = list(brain.biases)
    time_constants = list(brain.time_constants)
    states = list(brain.states)
    while len(biases) < target_hidden:
        biases.append(0.0)
    while len(time_constants) < target_hidden:
        time_constants.append(1.0)
    while len(states) < target_hidden:
        states.append(0.0)

    return GenomeBrainGene(
        input_weights=tuple(tuple(row) for row in input_weights),
        recurrent_weights=tuple(tuple(row) for row in recurrent_weights),
        biases=tuple(biases),
        time_constants=tuple(time_constants),
        output_size=target_output_size,
        states=tuple(states),
    )


def _resize_brain_hidden_neurons(
    brain: GenomeBrainGene,
    target_hidden_neurons: int,
) -> GenomeBrainGene:
    current_total = len(brain.biases)
    current_hidden = max(0, current_total - brain.output_size)
    target_hidden = max(0, target_hidden_neurons)
    target_total = target_hidden + brain.output_size
    input_size = len(brain.input_weights[0]) if brain.input_weights else 0

    current_rows = [list(row) for row in brain.input_weights]
    current_recurrent = [list(row) for row in brain.recurrent_weights]
    current_biases = list(brain.biases)
    current_time_constants = list(brain.time_constants)
    current_states = list(brain.states)

    hidden_rows = current_rows[:current_hidden]
    output_rows = current_rows[current_hidden:]
    hidden_biases = current_biases[:current_hidden]
    output_biases = current_biases[current_hidden:]
    hidden_taus = current_time_constants[:current_hidden]
    output_taus = current_time_constants[current_hidden:]
    hidden_states = current_states[:current_hidden]
    output_states = current_states[current_hidden:]

    if target_hidden > current_hidden:
        growth = target_hidden - current_hidden
        hidden_rows.extend([[0.0] * input_size for _ in range(growth)])
        hidden_biases.extend(0.0 for _ in range(growth))
        hidden_taus.extend(1.0 for _ in range(growth))
        hidden_states.extend(0.0 for _ in range(growth))
    elif target_hidden < current_hidden:
        shrink = current_hidden - target_hidden
        del hidden_rows[:shrink]
        del hidden_biases[:shrink]
        del hidden_taus[:shrink]
        del hidden_states[:shrink]

    rows = hidden_rows + output_rows
    biases = hidden_biases + output_biases
    time_constants = hidden_taus + output_taus
    states = hidden_states + output_states

    if len(rows) < target_total:
        rows.extend([[0.0] * input_size for _ in range(target_total - len(rows))])
    if len(biases) < target_total:
        biases.extend(0.0 for _ in range(target_total - len(biases)))
    if len(time_constants) < target_total:
        time_constants.extend(1.0 for _ in range(target_total - len(time_constants)))
    if len(states) < target_total:
        states.extend(0.0 for _ in range(target_total - len(states)))

    current_hidden_rows = current_recurrent[:current_hidden]
    current_output_rows = current_recurrent[current_hidden:]

    def _resize_recurrent_row(row: list[float]) -> list[float]:
        hidden_prefix = row[:current_hidden]
        output_suffix = row[current_hidden:]
        if target_hidden > current_hidden:
            hidden_prefix = hidden_prefix + [0.0] * (target_hidden - current_hidden)
        elif target_hidden < current_hidden:
            hidden_prefix = hidden_prefix[current_hidden - target_hidden :]
        resized = hidden_prefix + output_suffix
        if len(resized) < target_total:
            resized.extend(0.0 for _ in range(target_total - len(resized)))
        return resized[:target_total]

    resized_hidden_rows = [_resize_recurrent_row(row) for row in current_hidden_rows]
    resized_output_rows = [_resize_recurrent_row(row) for row in current_output_rows]
    if target_hidden > current_hidden:
        resized_hidden_rows.extend([[0.0] * target_total for _ in range(target_hidden - current_hidden)])
    elif target_hidden < current_hidden:
        resized_hidden_rows = resized_hidden_rows[current_hidden - target_hidden :]

    updated_recurrent = resized_hidden_rows + resized_output_rows

    return GenomeBrainGene(
        input_weights=tuple(tuple(row) for row in rows[:target_total]),
        recurrent_weights=tuple(tuple(row) for row in updated_recurrent[:target_total]),
        biases=tuple(biases[:target_total]),
        time_constants=tuple(time_constants[:target_total]),
        output_size=brain.output_size,
        states=tuple(states[:target_total]),
    )


def genome_distance(left: CreatureGenome | None, right: CreatureGenome | None) -> float:
    if left is None and right is None:
        return 0.0
    if left is None or right is None:
        return 1_000.0

    distance = 0.0
    distance += abs(len(left.nodes) - len(right.nodes))
    distance += abs(len(left.edges) - len(right.edges))

    for node_type in MUTABLE_NODE_TYPES:
        distance += abs(
            sum(1 for node in left.nodes if node.node_type == node_type)
            - sum(1 for node in right.nodes if node.node_type == node_type)
        )

    distance += abs(sum(1 for edge in left.edges if edge.has_motor) - sum(1 for edge in right.edges if edge.has_motor))

    if left.nodes and right.nodes:
        left_mean_radius = sum(node.radius for node in left.nodes) / len(left.nodes)
        right_mean_radius = sum(node.radius for node in right.nodes) / len(right.nodes)
        distance += abs(left_mean_radius - right_mean_radius)
    distance += abs(left.visuals.silhouette_scale - right.visuals.silhouette_scale) * 0.2
    distance += abs(left.visuals.glyph_scale - right.visuals.glyph_scale) * 0.15
    distance += abs(left.visuals.band_count - right.visuals.band_count) * 0.1

    if left.brain is None and right.brain is None:
        return distance
    if left.brain is None or right.brain is None:
        return distance + 5.0

    distance += abs(left.brain.output_size - right.brain.output_size) * 0.5
    if len(left.brain.biases) == len(right.brain.biases):
        distance += (
            sum(abs(a - b) for a, b in zip(left.brain.biases, right.brain.biases, strict=True))
            / max(len(left.brain.biases), 1)
        ) * 0.1
        distance += (
            sum(abs(a - b) for a, b in zip(left.brain.time_constants, right.brain.time_constants, strict=True))
            / max(len(left.brain.time_constants), 1)
        ) * 0.05
    return distance


def cluster_species(
    genomes: tuple[CreatureGenome | None, ...],
    threshold: float = 1.5,
) -> tuple[str, ...]:
    labels = [""] * len(genomes)

    for start_index in range(len(genomes)):
        if labels[start_index]:
            continue
        representative_hash = genome_hash(genomes[start_index]) or f"none-{start_index:03d}"
        label = f"species-{representative_hash}"
        labels[start_index] = label
        stack = [start_index]
        while stack:
            current = stack.pop()
            for other in range(len(genomes)):
                if labels[other]:
                    continue
                if genome_distance(genomes[current], genomes[other]) <= threshold:
                    labels[other] = label
                    stack.append(other)

    return tuple(labels)


def encode_creature_genome(
    *,
    nodes: list[NodeState],
    edges: list[EdgeState],
    creature: CreatureState,
) -> CreatureGenome:
    if not creature.node_indices:
        return CreatureGenome(nodes=(), edges=(), brain=None, color_rgb=creature.color_rgb)

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
    return CreatureGenome(
        nodes=genome_nodes,
        edges=genome_edges,
        brain=genome_brain,
        color_rgb=creature.color_rgb,
    )


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
    motor_toggle_mutation_rate: float = 0.0,
    node_type_mutation_rate: float = 0.0,
    structural_mutation_rate: float = 0.0,
    hidden_neuron_mutation_rate: float = 0.0,
    max_hidden_neurons: int = 24,
    chain_extension_mutation_rate: float = 0.0,
    max_nodes_per_creature: int = 16,
    color_sigma: float = 6.0,
) -> CreatureGenome:
    mutated_nodes = [
        GenomeNodeGene(
            position=node.position
            + Vec2(
                rng.gauss(0.0, position_sigma),
                rng.gauss(0.0, position_sigma),
            ),
            radius=max(0.1, node.radius + rng.gauss(0.0, radius_sigma)),
            node_type=(
                _mutated_node_type(node.node_type, rng)
                if rng.random() < node_type_mutation_rate
                else node.node_type
            ),
        )
        for node in genome.nodes
    ]
    mutated_edges = [
        GenomeEdgeGene(
            a=edge.a,
            b=edge.b,
            rest_length=edge.rest_length,
            stiffness=edge.stiffness,
            has_motor=(not edge.has_motor) if rng.random() < motor_toggle_mutation_rate else edge.has_motor,
            motor_strength=0.0,
        )
        for edge in genome.edges
    ]
    mutated_edges = [
        GenomeEdgeGene(
            a=edge.a,
            b=edge.b,
            rest_length=edge.rest_length,
            stiffness=edge.stiffness,
            has_motor=edge.has_motor,
            motor_strength=(
                max(0.5, abs(rng.gauss(1.0, max(motor_strength_sigma, 0.1))))
                if edge.has_motor and not original_edge.has_motor
                else max(0.0, original_edge.motor_strength + rng.gauss(0.0, motor_strength_sigma))
                if edge.has_motor
                else 0.0
            ),
        )
        for edge, original_edge in zip(mutated_edges, genome.edges, strict=True)
    ]
    if mutated_nodes and len(mutated_nodes) < max_nodes_per_creature and rng.random() < structural_mutation_rate:
        parent_index = rng.randrange(len(mutated_nodes))
        parent_node = mutated_nodes[parent_index]
        offset = Vec2(rng.uniform(-3.0, 3.0), rng.uniform(-3.0, 3.0))
        new_node = GenomeNodeGene(
            position=parent_node.position + offset,
            radius=max(0.1, parent_node.radius + rng.gauss(0.0, max(radius_sigma, 0.01))),
            node_type=rng.choice(MUTABLE_NODE_TYPES),
        )
        new_node_index = len(mutated_nodes)
        mutated_nodes.append(new_node)
        mutated_edges.append(
            GenomeEdgeGene(
                a=parent_index,
                b=new_node_index,
                rest_length=max(offset.magnitude(), 0.1),
                stiffness=1.0,
                has_motor=False,
                motor_strength=0.0,
            )
        )
    chain_extension_parent_motor_idx: int | None = None
    chain_extension_new_motor_idx: int | None = None
    if (
        len(mutated_nodes) >= 2
        and len(mutated_nodes) < max_nodes_per_creature
        and rng.random() < chain_extension_mutation_rate
    ):
        degree: dict[int, int] = defaultdict(int)
        for edge in mutated_edges:
            degree[edge.a] += 1
            degree[edge.b] += 1
        terminals = [i for i in range(len(mutated_nodes)) if degree.get(i, 0) == 1]
        if terminals:
            terminal_idx = rng.choice(terminals)
            terminal_edge = next(
                edge for edge in mutated_edges if edge.a == terminal_idx or edge.b == terminal_idx
            )
            neighbor_idx = terminal_edge.b if terminal_edge.a == terminal_idx else terminal_edge.a
            terminal_node = mutated_nodes[terminal_idx]
            neighbor_node = mutated_nodes[neighbor_idx]
            dx = terminal_node.position.x - neighbor_node.position.x
            dy = terminal_node.position.y - neighbor_node.position.y
            length = math.hypot(dx, dy)
            if length < 1e-9:
                dx, dy = rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)
                length = math.hypot(dx, dy) or 1.0
            dx /= length
            dy /= length
            angle_noise = rng.gauss(0.0, 0.3)
            cos_a, sin_a = math.cos(angle_noise), math.sin(angle_noise)
            dx2 = dx * cos_a - dy * sin_a
            dy2 = dx * sin_a + dy * cos_a
            extend_length = max(0.5, terminal_edge.rest_length * rng.uniform(0.8, 1.2))
            offset = Vec2(dx2 * extend_length, dy2 * extend_length)
            new_node = GenomeNodeGene(
                position=terminal_node.position + offset,
                radius=terminal_node.radius,
                node_type=terminal_node.node_type,
            )
            new_node_index = len(mutated_nodes)
            mutated_nodes.append(new_node)
            new_edge = GenomeEdgeGene(
                a=terminal_idx,
                b=new_node_index,
                rest_length=max(offset.magnitude(), 0.1),
                stiffness=1.0,
                has_motor=True,
                motor_strength=max(0.5, rng.gauss(1.5, 0.3)),
            )
            mutated_edges.append(new_edge)
            # Find motor index of parent edge for brain warm-start
            motor_count = 0
            for edge in mutated_edges[:-1]:
                if edge.has_motor:
                    if edge is terminal_edge:
                        chain_extension_parent_motor_idx = motor_count
                    motor_count += 1
            chain_extension_new_motor_idx = sum(1 for e in mutated_edges if e.has_motor) - 1
    mutated_genome_without_brain = CreatureGenome(
        nodes=tuple(mutated_nodes),
        edges=tuple(mutated_edges),
        brain=None,
        color_rgb=genome.color_rgb,
        visuals=genome.visuals,
    )
    mutated_brain = None
    if genome.brain is not None:
        original_surplus_outputs = max(0, genome.brain.output_size - required_control_outputs(genome))
        target_output_size = required_control_outputs(mutated_genome_without_brain) + original_surplus_outputs
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
        mutated_brain = _resize_brain_for_outputs(mutated_brain, target_output_size)
        current_hidden = max(0, len(mutated_brain.biases) - mutated_brain.output_size)
        if hidden_neuron_mutation_rate > 0.0 and rng.random() < hidden_neuron_mutation_rate:
            if current_hidden <= 0:
                target_hidden = 1
            elif current_hidden >= max_hidden_neurons:
                target_hidden = current_hidden - 1
            else:
                target_hidden = current_hidden + 1 if rng.random() < 0.5 else current_hidden - 1
            mutated_brain = _resize_brain_hidden_neurons(
                mutated_brain,
                target_hidden_neurons=min(max_hidden_neurons, max(0, target_hidden)),
            )
        if (
            chain_extension_parent_motor_idx is not None
            and chain_extension_new_motor_idx is not None
            and mutated_brain is not None
        ):
            mutated_brain = _warm_start_chain_neuron(
                mutated_brain, chain_extension_parent_motor_idx, chain_extension_new_motor_idx, rng
            )
    return CreatureGenome(
        nodes=tuple(mutated_nodes),
        edges=tuple(mutated_edges),
        brain=mutated_brain,
        color_rgb=_mutate_color_rgb(genome.color_rgb, rng=rng, sigma=color_sigma),
        visuals=_mutate_visuals(genome.visuals, rng=rng),
    )


def _warm_start_chain_neuron(
    brain: GenomeBrainGene,
    parent_motor_idx: int,
    new_motor_idx: int,
    rng: random.Random,
) -> GenomeBrainGene:
    """Bootstrap phase-shifted coordination for a chain-extension motor neuron."""
    n = len(brain.biases)
    hidden_count = max(0, n - brain.output_size)
    parent_neuron_idx = hidden_count + parent_motor_idx
    new_neuron_idx = hidden_count + new_motor_idx
    if parent_neuron_idx >= n or new_neuron_idx >= n:
        return brain
    coupling_weight = rng.uniform(0.2, 0.6)
    recurrent = [list(row) for row in brain.recurrent_weights]
    recurrent[new_neuron_idx][parent_neuron_idx] = coupling_weight
    parent_tau = brain.time_constants[parent_neuron_idx]
    new_tau = parent_tau * rng.uniform(1.1, 1.5)
    time_constants = list(brain.time_constants)
    time_constants[new_neuron_idx] = new_tau
    return GenomeBrainGene(
        input_weights=brain.input_weights,
        recurrent_weights=tuple(tuple(row) for row in recurrent),
        biases=brain.biases,
        time_constants=tuple(time_constants),
        output_size=brain.output_size,
        states=brain.states,
    )


def genome_to_dict(genome: CreatureGenome | None) -> dict[str, Any] | None:
    if genome is None:
        return None
    return {
        "color_rgb": list(genome.color_rgb),
        "visuals": {
            "silhouette_scale": genome.visuals.silhouette_scale,
            "glyph_scale": genome.visuals.glyph_scale,
            "band_count": genome.visuals.band_count,
            "band_offset": genome.visuals.band_offset,
        },
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
    color_rgb = tuple(payload.get("color_rgb", DEFAULT_LINEAGE_COLOR_RGB))
    visuals_payload = payload.get("visuals", {})
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
        color_rgb=tuple(_clamp_rgb_channel(channel) for channel in color_rgb),
        visuals=GenomeVisualGene(
            silhouette_scale=float(visuals_payload.get("silhouette_scale", 1.0)),
            glyph_scale=float(visuals_payload.get("glyph_scale", 1.0)),
            band_count=int(visuals_payload.get("band_count", 2)),
            band_offset=float(visuals_payload.get("band_offset", 0.0)),
        ),
    )


def genome_hash(genome: CreatureGenome | None) -> str:
    if genome is None:
        return ""
    canonical = json.dumps(genome_to_dict(genome), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]


def coarse_species_signature(genome: CreatureGenome | None) -> str:
    if genome is None:
        return ""
    mouths = sum(1 for node in genome.nodes if node.node_type == NodeType.MOUTH)
    grippers = sum(1 for node in genome.nodes if node.node_type == NodeType.GRIPPER)
    sensors = sum(1 for node in genome.nodes if node.node_type == NodeType.SENSOR)
    photoreceptors = sum(1 for node in genome.nodes if node.node_type == NodeType.PHOTORECEPTOR)
    motors = sum(1 for edge in genome.edges if edge.has_motor)
    mean_radius = round(sum(node.radius for node in genome.nodes) / max(len(genome.nodes), 1), 1)
    return "|".join(
        [
            f"n{len(genome.nodes)}",
            f"e{len(genome.edges)}",
            f"m{mouths}",
            f"g{grippers}",
            f"s{sensors}",
            f"p{photoreceptors}",
            f"mot{motors}",
            f"r{mean_radius}",
        ]
    )
