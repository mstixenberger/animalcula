"""Mutation helpers for inherited world state."""

from __future__ import annotations

from dataclasses import replace
import random

from animalcula.sim.types import BrainState, NodeState, Vec2


def mutate_node(
    node: NodeState,
    rng: random.Random,
    position_sigma: float,
    radius_sigma: float,
) -> NodeState:
    mutated_position = node.position + Vec2(
        rng.gauss(0.0, position_sigma),
        rng.gauss(0.0, position_sigma),
    )
    mutated_radius = max(0.1, node.radius + rng.gauss(0.0, radius_sigma))
    return replace(node, position=mutated_position, radius=mutated_radius)


def mutate_brain(
    brain: BrainState,
    rng: random.Random,
    weight_sigma: float,
    bias_sigma: float,
    tau_sigma: float,
) -> BrainState:
    input_weights = tuple(
        tuple(weight + rng.gauss(0.0, weight_sigma) for weight in row)
        for row in brain.input_weights
    )
    recurrent_weights = tuple(
        tuple(weight + rng.gauss(0.0, weight_sigma) for weight in row)
        for row in brain.recurrent_weights
    )
    biases = tuple(bias + rng.gauss(0.0, bias_sigma) for bias in brain.biases)
    time_constants = tuple(
        max(0.1, tau + rng.gauss(0.0, tau_sigma)) for tau in brain.time_constants
    )
    return replace(
        brain,
        input_weights=input_weights,
        recurrent_weights=recurrent_weights,
        biases=biases,
        time_constants=time_constants,
    )
