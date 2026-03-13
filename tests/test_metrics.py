import math

from animalcula.analysis.metrics import (
    interestingness_score,
    shannon_diversity,
    trophic_balance_score,
    trophic_percentages,
)


def test_interestingness_score_rewards_nonzero_population_and_energy() -> None:
    low = interestingness_score(population=0, total_energy=0.0, births=0, deaths=0, reproductions=0)
    high = interestingness_score(population=6, total_energy=3.0, births=4, deaths=1, reproductions=4)

    assert high > low
    assert low == 0.0


def test_interestingness_score_rewards_lifecycle_activity() -> None:
    quiet = interestingness_score(population=3, total_energy=3.0, births=0, deaths=0, reproductions=0)
    active = interestingness_score(population=3, total_energy=3.0, births=5, deaths=2, reproductions=5)

    assert active > quiet


def test_interestingness_score_penalizes_runaway_dominance() -> None:
    diverse = interestingness_score(
        population=10,
        total_energy=10.0,
        births=4,
        deaths=1,
        reproductions=4,
        species_turnover=2,
        observed_species_count=4,
        peak_species_fraction=0.55,
        runaway_dominance_detected=False,
    )
    monoculture = interestingness_score(
        population=10,
        total_energy=10.0,
        births=4,
        deaths=1,
        reproductions=4,
        species_turnover=2,
        observed_species_count=4,
        peak_species_fraction=0.95,
        runaway_dominance_detected=True,
    )

    assert diverse > monoculture


def test_interestingness_score_rewards_midrange_capacity_fraction() -> None:
    low = interestingness_score(
        population=10,
        total_energy=5.0,
        births=1,
        deaths=1,
        reproductions=1,
        population_capacity_fraction=0.05,
    )
    mid = interestingness_score(
        population=10,
        total_energy=5.0,
        births=1,
        deaths=1,
        reproductions=1,
        population_capacity_fraction=0.55,
    )
    saturated = interestingness_score(
        population=10,
        total_energy=5.0,
        births=1,
        deaths=1,
        reproductions=1,
        population_capacity_fraction=1.0,
    )

    assert mid > low
    assert mid > saturated


def test_shannon_diversity_is_zero_for_single_lineage() -> None:
    assert shannon_diversity({"aaa": 5}) == 0.0


def test_shannon_diversity_increases_with_more_even_lineages() -> None:
    uneven = shannon_diversity({"aaa": 9, "bbb": 1})
    even = shannon_diversity({"aaa": 5, "bbb": 5})

    assert even > uneven


def test_trophic_percentages_normalize_counts() -> None:
    trophic = trophic_percentages(autotrophs=2, herbivores=1, predators=1)

    assert trophic["autotrophs"] == 0.5
    assert trophic["herbivores"] == 0.25
    assert trophic["predators"] == 0.25


def test_trophic_balance_score_rewards_more_even_role_mix() -> None:
    imbalanced = trophic_balance_score(autotrophs=3, herbivores=0, predators=0)
    balanced = trophic_balance_score(autotrophs=1, herbivores=1, predators=1)

    assert balanced > imbalanced
    assert math.isclose(balanced, 1.0)
