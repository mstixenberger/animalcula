from animalcula.analysis.metrics import interestingness_score


def test_interestingness_score_rewards_nonzero_population_and_energy() -> None:
    low = interestingness_score(population=0, total_energy=0.0)
    high = interestingness_score(population=6, total_energy=3.0)

    assert high > low
    assert low == 0.0
