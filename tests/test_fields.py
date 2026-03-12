from pathlib import Path

from animalcula import Config, World
from animalcula.sim.fields import Grid2D
from animalcula.sim.types import Vec2


def test_grid_dimensions_follow_world_size_and_resolution() -> None:
    grid = Grid2D(width=10.0, height=6.0, resolution=2.0)

    assert grid.cols == 5
    assert grid.rows == 3


def test_grid_sampling_wraps_toroidal_coordinates() -> None:
    grid = Grid2D(width=10.0, height=10.0, resolution=5.0)
    grid.set_value(col=0, row=1, value=7.0)

    assert grid.sample(Vec2(10.1, 7.5)) == 7.0


def test_world_initializes_light_grid_from_config() -> None:
    config = Config.from_yaml(Path("config/default.yaml"))
    world = World(config=config)

    assert world.light_grid.cols == 200
    assert world.light_grid.rows == 200


def test_world_light_grid_forms_a_directional_gradient() -> None:
    config = Config.from_yaml(Path("config/default.yaml"))
    world = World(config=config)

    left = world.light_grid.sample(Vec2(5.0, 500.0))
    right = world.light_grid.sample(Vec2(995.0, 500.0))

    assert right > left


def test_world_initializes_positive_nutrient_sources() -> None:
    config = Config.from_yaml(Path("config/default.yaml"))
    world = World(config=config)

    assert any(value > 0.0 for value in world.nutrient_grid.values)


def test_nutrient_source_placement_is_deterministic_for_seed() -> None:
    config = Config.from_yaml(Path("config/default.yaml"))
    world_a = World(config=config, seed=123)
    world_b = World(config=config, seed=123)

    assert world_a.nutrient_grid.values == world_b.nutrient_grid.values
