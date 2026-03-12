"""Run-quality heuristics for headless exploration."""

from __future__ import annotations

import math


def interestingness_score(
    population: int,
    total_energy: float,
    births: int,
    deaths: int,
    reproductions: int,
    predation_kills: int = 0,
) -> float:
    if population <= 0:
        return 0.0
    return (
        float(population)
        + max(total_energy, 0.0)
        + (0.5 * births)
        + deaths
        + reproductions
        + (2.0 * predation_kills)
    )


def shannon_diversity(lineage_counts: dict[str, int]) -> float:
    total = sum(lineage_counts.values())
    if total <= 0:
        return 0.0

    entropy = 0.0
    for count in lineage_counts.values():
        if count <= 0:
            continue
        proportion = count / total
        entropy -= proportion * math.log(proportion)
    return entropy
