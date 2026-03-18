"""Energy accounting helpers."""

from __future__ import annotations

from animalcula.sim.types import Vec2


def basal_cost(node_count: int, basal_cost_per_node: float) -> float:
    return node_count * basal_cost_per_node


def feeding_gain(nutrient_level: float, mouth_count: int, feed_rate: float) -> float:
    return nutrient_level * mouth_count * feed_rate


def photosynthesis_gain(
    light_level: float,
    receptor_count: int,
    photosynthesis_rate: float,
) -> float:
    return light_level * receptor_count * photosynthesis_rate


def motor_cost(total_actuation: float, motor_cost_per_unit: float) -> float:
    return total_actuation * motor_cost_per_unit


def reach_multiplier(node_pos: Vec2, com: Vec2, max_extent: float, bonus: float) -> float:
    if max_extent <= 0.0 or bonus <= 0.0:
        return 1.0
    dist = (node_pos - com).magnitude()
    return 1.0 + bonus * min(1.0, dist / max_extent)
