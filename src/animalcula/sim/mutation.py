"""Mutation helpers for inherited world state."""

from __future__ import annotations

from dataclasses import replace
import random

from animalcula.sim.types import NodeState, Vec2


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
