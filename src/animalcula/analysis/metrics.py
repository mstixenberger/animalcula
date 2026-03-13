"""Run-quality heuristics for headless exploration."""

from __future__ import annotations

import math


def interestingness_score(
    population: int,
    total_energy: float,
    births: int,
    deaths: int,
    reproductions: int,
    speciation_events: int = 0,
    species_extinctions: int = 0,
    predation_kills: int = 0,
    species_turnover: int = 0,
    observed_species_count: int = 0,
    peak_species_fraction: float = 0.0,
    runaway_dominance_detected: bool = False,
    population_capacity_fraction: float = 0.0,
    trophic_balance: float = 0.0,
) -> float:
    if population <= 0:
        return 0.0
    score = (
        float(population)
        + max(total_energy, 0.0)
        + (0.5 * births)
        + deaths
        + reproductions
        + (1.5 * speciation_events)
        + (0.5 * species_extinctions)
        + (2.0 * predation_kills)
        + species_turnover
        + (0.5 * observed_species_count)
    )
    dominance_penalty = max(0.0, peak_species_fraction - 0.5) * 10.0
    if runaway_dominance_detected:
        dominance_penalty += 10.0
    capacity_bonus = max(0.0, 1.0 - (abs(population_capacity_fraction - 0.55) / 0.55)) * 5.0
    trophic_bonus = max(0.0, trophic_balance) * 5.0
    return max(0.0, score + capacity_bonus + trophic_bonus - dominance_penalty)


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


def trophic_percentages(autotrophs: int, herbivores: int, predators: int) -> dict[str, float]:
    total = autotrophs + herbivores + predators
    if total <= 0:
        return {
            "autotrophs": 0.0,
            "herbivores": 0.0,
            "predators": 0.0,
        }
    return {
        "autotrophs": autotrophs / total,
        "herbivores": herbivores / total,
        "predators": predators / total,
    }


def trophic_balance_score(autotrophs: int, herbivores: int, predators: int) -> float:
    percentages = trophic_percentages(autotrophs=autotrophs, herbivores=herbivores, predators=predators)
    total = autotrophs + herbivores + predators
    if total <= 0:
        return 0.0

    entropy = 0.0
    for proportion in percentages.values():
        if proportion <= 0.0:
            continue
        entropy -= proportion * math.log(proportion)
    return entropy / math.log(3.0)
