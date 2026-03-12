import math

from animalcula.sim.energy import basal_cost, feeding_gain, motor_cost, photosynthesis_gain


def test_basal_cost_scales_with_node_count() -> None:
    assert math.isclose(basal_cost(node_count=3, basal_cost_per_node=0.25), 0.75)


def test_photosynthesis_gain_scales_with_light_and_receptors() -> None:
    gain = photosynthesis_gain(
        light_level=0.8,
        receptor_count=2,
        photosynthesis_rate=0.5,
    )

    assert math.isclose(gain, 0.8)


def test_feeding_gain_scales_with_nutrients_and_mouths() -> None:
    gain = feeding_gain(
        nutrient_level=1.5,
        mouth_count=2,
        feed_rate=0.25,
    )

    assert math.isclose(gain, 0.75)


def test_motor_cost_scales_with_applied_actuation() -> None:
    cost = motor_cost(total_actuation=3.0, motor_cost_per_unit=0.25)

    assert math.isclose(cost, 0.75)
