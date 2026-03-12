from pathlib import Path

from animalcula import Config, World


def test_world_checkpoint_roundtrip_preserves_state(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "world.json"
    config = Config.from_yaml(Path("config/default.yaml"))
    world = World(config=config, seed=7)
    world.seed_demo_archetypes()
    world.step(3)

    world.save(checkpoint_path)
    restored = World.load(checkpoint_path)

    assert restored.tick == world.tick
    assert restored.seed == world.seed
    assert restored.nodes == world.nodes
    assert restored.edges == world.edges
    assert restored.creatures == world.creatures
    assert restored.nutrient_grid.values == world.nutrient_grid.values
    assert restored.light_grid.values == world.light_grid.values
    assert restored.chemical_a_grid.values == world.chemical_a_grid.values
    assert restored.chemical_b_grid.values == world.chemical_b_grid.values
    assert restored.events == world.events
