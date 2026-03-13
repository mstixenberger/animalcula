from pathlib import Path

import pytest

from animalcula import Config


def test_loads_default_config() -> None:
    config = Config.from_yaml(Path("config/default.yaml"))

    assert config.world.width == 1000.0
    assert config.world.boundary == "toroidal"
    assert config.physics.dt == 0.01
    assert config.physics.grip_spring_stiffness == 200.0
    assert config.physics.grip_yield_force == 50.0
    assert config.environment.chemical_diffusion_rate == 0.2
    assert config.environment.chemical_decay_rate == 0.05
    assert config.environment.nutrient_shift_interval == 1000
    assert config.environment.nutrient_shift_count == 1
    assert config.environment.light_intensity_max == 1.0
    assert config.environment.light_intensity_min == 0.5
    assert config.environment.light_direction == (1.0, 0.0)
    assert config.environment.light_season_interval == 10000
    assert config.environment.light_season_steps == 4
    assert config.energy.basal_cost_per_node == 0.001
    assert config.energy.feed_rate == 0.01
    assert config.energy.grip_cost == 0.002
    assert config.energy.photosynthesis_rate == 0.005
    assert config.energy.reproduction_threshold == 100.0
    assert config.evolution.position_mutation_sigma == 0.5
    assert config.evolution.radius_mutation_sigma == 0.05
    assert config.evolution.weight_mutation_sigma == 0.1
    assert config.evolution.bias_mutation_sigma == 0.05
    assert config.evolution.tau_mutation_sigma == 0.02
    assert config.evolution.motor_strength_mutation_sigma == 0.2
    assert config.evolution.motor_toggle_mutation_rate == 0.05
    assert config.evolution.node_type_mutation_rate == 0.02
    assert config.evolution.structural_mutation_rate == 0.05
    assert config.evolution.hidden_neuron_mutation_rate == 0.05
    assert config.evolution.max_hidden_neurons == 24
    assert config.brain.motor_force_scale == 1.0
    assert config.brain.default_input_size == 16
    assert config.creatures.min_population == 0
    assert config.creatures.max_population == 500
    assert config.simulation.initial_seed == 42


def test_rejects_non_mapping_root(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("- not-a-mapping\n", encoding="utf-8")

    with pytest.raises(TypeError, match="configuration root must be a mapping"):
        Config.from_yaml(config_path)


def test_config_can_apply_nested_overrides() -> None:
    config = Config.from_yaml(Path("config/default.yaml"))

    updated = config.with_overrides(
        [
            "energy.reproduction_threshold=0.1",
            "environment.nutrient_source_strength=3.5",
        ]
    )

    assert updated.energy.reproduction_threshold == 0.1
    assert updated.environment.nutrient_source_strength == 3.5


def test_loads_nursery_config_profile() -> None:
    config = Config.from_yaml(Path("config/nursery.yaml"))

    assert config.energy.basal_cost_per_node < 0.001
    assert config.environment.nutrient_source_strength > 2.0
    assert config.evolution.motor_toggle_mutation_rate > 0.0
    assert config.evolution.node_type_mutation_rate > 0.0
    assert config.evolution.structural_mutation_rate > 0.0
    assert config.evolution.hidden_neuron_mutation_rate > 0.0
    assert config.evolution.max_hidden_neurons >= 8
    assert config.creatures.min_population == 0


def test_config_from_dict_backfills_hidden_neuron_defaults_for_old_payloads() -> None:
    config = Config.from_yaml(Path("config/default.yaml"))
    payload = config.to_dict()
    del payload["evolution"]["hidden_neuron_mutation_rate"]
    del payload["evolution"]["max_hidden_neurons"]

    loaded = Config.from_dict(payload)

    assert loaded.evolution.hidden_neuron_mutation_rate == 0.0
    assert loaded.evolution.max_hidden_neurons == 24


def test_config_from_dict_backfills_nutrient_shift_defaults_for_old_payloads() -> None:
    config = Config.from_yaml(Path("config/default.yaml"))
    payload = config.to_dict()
    del payload["environment"]["nutrient_shift_interval"]
    del payload["environment"]["nutrient_shift_count"]

    loaded = Config.from_dict(payload)

    assert loaded.environment.nutrient_shift_interval == 0
    assert loaded.environment.nutrient_shift_count == 1


def test_config_from_dict_backfills_light_season_defaults_for_old_payloads() -> None:
    config = Config.from_yaml(Path("config/default.yaml"))
    payload = config.to_dict()
    del payload["environment"]["light_intensity_min"]
    del payload["environment"]["light_season_interval"]
    del payload["environment"]["light_season_steps"]

    loaded = Config.from_dict(payload)

    assert loaded.environment.light_intensity_min == config.environment.light_intensity_max
    assert loaded.environment.light_season_interval == 0
    assert loaded.environment.light_season_steps == 1
