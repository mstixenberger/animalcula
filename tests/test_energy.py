import math

from animalcula.sim.energy import basal_cost, photosynthesis_gain


def test_basal_cost_scales_with_node_count() -> None:
    assert math.isclose(basal_cost(node_count=3, basal_cost_per_node=0.25), 0.75)


def test_photosynthesis_gain_scales_with_light_and_receptors() -> None:
    gain = photosynthesis_gain(
        light_level=0.8,
        receptor_count=2,
        photosynthesis_rate=0.5,
    )

    assert math.isclose(gain, 0.8)
