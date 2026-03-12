"""Run-quality heuristics for headless exploration."""

from __future__ import annotations


def interestingness_score(population: int, total_energy: float) -> float:
    if population <= 0:
        return 0.0
    return float(population) + max(total_energy, 0.0)
