"""Energy accounting helpers."""

from __future__ import annotations


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
