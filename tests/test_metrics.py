from animalcula.analysis.metrics import interestingness_score, shannon_diversity


def test_interestingness_score_rewards_nonzero_population_and_energy() -> None:
    low = interestingness_score(population=0, total_energy=0.0, births=0, deaths=0, reproductions=0)
    high = interestingness_score(population=6, total_energy=3.0, births=4, deaths=1, reproductions=4)

    assert high > low
    assert low == 0.0


def test_interestingness_score_rewards_lifecycle_activity() -> None:
    quiet = interestingness_score(population=3, total_energy=3.0, births=0, deaths=0, reproductions=0)
    active = interestingness_score(population=3, total_energy=3.0, births=5, deaths=2, reproductions=5)

    assert active > quiet


def test_shannon_diversity_is_zero_for_single_lineage() -> None:
    assert shannon_diversity({"aaa": 5}) == 0.0


def test_shannon_diversity_increases_with_more_even_lineages() -> None:
    uneven = shannon_diversity({"aaa": 9, "bbb": 1})
    even = shannon_diversity({"aaa": 5, "bbb": 5})

    assert even > uneven
