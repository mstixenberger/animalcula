from pathlib import Path

import pytest

from animalcula import Config


def test_loads_default_config() -> None:
    config = Config.from_yaml(Path("config/default.yaml"))

    assert config.world.width == 1000.0
    assert config.world.boundary == "toroidal"
    assert config.physics.dt == 0.01
    assert config.simulation.initial_seed == 42


def test_rejects_non_mapping_root(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("- not-a-mapping\n", encoding="utf-8")

    with pytest.raises(TypeError, match="configuration root must be a mapping"):
        Config.from_yaml(config_path)
