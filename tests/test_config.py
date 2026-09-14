"""Unit tests for configuration loading utility."""

import json
from pathlib import Path
import pytest
from src.common.config import load_config


class TestConfigLoader:
    """Tests for load_config utility with JSON and YAML support."""

    def test_load_existing_json_config(self, tmp_path: Path):
        """Verify existing JSON configuration loading behavior."""
        json_file = tmp_path / "test_config.json"
        data = {
            "game": "test_game",
            "seed": 123,
            "params": {"learning_rate": 0.01, "layers": [64, 64]},
        }
        json_file.write_text(json.dumps(data), encoding="utf-8")

        loaded = load_config(json_file)
        assert isinstance(loaded, dict)
        assert loaded["game"] == "test_game"
        assert loaded["seed"] == 123
        assert loaded["params"]["learning_rate"] == 0.01
        assert loaded["params"]["layers"] == [64, 64]

    def test_load_snake_dqn_yaml_config(self):
        """Verify configs/snake_dqn.yaml loads successfully."""
        yaml_path = Path("configs/snake_dqn.yaml")
        assert yaml_path.is_file(), "configs/snake_dqn.yaml must exist"

        config = load_config(yaml_path)
        assert isinstance(config, dict)
        assert config["game"] == "snake"
        assert config["agent"] == "refined_dqn"

    def test_snake_dqn_types_and_nested_values(self):
        """Verify important nested values are parsed with exact expected Python types."""
        config = load_config("configs/snake_dqn.yaml")

        # Top-level types
        assert isinstance(config["game"], str)
        assert isinstance(config["agent"], str)
        assert isinstance(config["seed"], int)

        # State types
        assert isinstance(config["state"], dict)
        assert isinstance(config["state"]["shape"], list)
        assert config["state"]["shape"] == [4, 64, 64]
        assert all(isinstance(x, int) for x in config["state"]["shape"])

        # Network types
        assert isinstance(config["network"], dict)
        assert isinstance(config["network"]["type"], str)
        assert isinstance(config["network"]["conv_channels"], list)
        assert config["network"]["conv_channels"] == [16, 32, 64]
        assert isinstance(config["network"]["fully_connected_units"], int)

        # Replay types
        assert isinstance(config["replay"], dict)
        assert isinstance(config["replay"]["high_error_capacity"], int)
        assert isinstance(config["replay"]["low_error_capacity"], int)
        assert isinstance(config["replay"]["total_capacity"], int)
        assert isinstance(config["replay"]["td_threshold"], float)
        assert isinstance(config["replay"]["high_error_fraction"], float)
        assert isinstance(config["replay"]["low_error_fraction"], float)

        # Training types
        assert isinstance(config["training"], dict)
        assert isinstance(config["training"]["batch_size"], int)
        assert isinstance(config["training"]["learning_rate"], float)
        assert isinstance(config["training"]["gamma"], float)
        assert isinstance(config["training"]["target_update_steps"], int)
        assert isinstance(config["training"]["training_gap"], int)

        # Epsilon types
        assert isinstance(config["epsilon"], dict)
        assert isinstance(config["epsilon"]["start"], float)
        assert isinstance(config["epsilon"]["end"], float)
        assert isinstance(config["epsilon"]["decay_frames"], int)

        # Checkpoint & Evaluation types
        assert isinstance(config["checkpoint"], dict)
        assert isinstance(config["checkpoint"]["interval_episodes"], int)
        assert isinstance(config["evaluation"], dict)
        assert isinstance(config["evaluation"]["episodes"], int)

    def test_snake_dqn_official_milestone3_values(self):
        """Verify Snake configuration contains official Milestone 3 values."""
        config = load_config("configs/snake_dqn.yaml")

        assert config["game"] == "snake"
        assert config["agent"] == "refined_dqn"
        assert config["state"]["shape"] == [4, 64, 64]
        assert config["network"]["type"] == "dueling_dqn"
        assert config["network"]["conv_channels"] == [16, 32, 64]
        assert config["network"]["fully_connected_units"] == 256
        assert config["replay"]["high_error_capacity"] == 35000
        assert config["replay"]["low_error_capacity"] == 15000
        assert config["replay"]["total_capacity"] == 50000
        assert config["replay"]["td_threshold"] == 0.5
        assert config["replay"]["high_error_fraction"] == 0.70
        assert config["replay"]["low_error_fraction"] == 0.30
        assert config["training"]["batch_size"] == 32
        assert config["training"]["learning_rate"] == 0.0001
        assert config["training"]["gamma"] == 0.99
        assert config["training"]["target_update_steps"] == 500
        assert config["training"]["training_gap"] == 4
        assert config["epsilon"]["start"] == 1.0
        assert config["epsilon"]["end"] == 0.02
        assert config["epsilon"]["decay_frames"] == 200000
        assert config["checkpoint"]["interval_episodes"] == 100
        assert config["evaluation"]["episodes"] == 100
        assert config["seed"] == 42

    def test_file_not_found(self):
        """Verify FileNotFoundError for missing config file."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            load_config("configs/non_existent_config.yaml")

    def test_invalid_yaml_raises_value_error(self, tmp_path: Path):
        """Verify ValueError is raised for malformed YAML."""
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("key: [unclosed list", encoding="utf-8")
        with pytest.raises(ValueError, match="Failed to parse YAML configuration"):
            load_config(bad_yaml)

    def test_base_config_merging(self):
        """Verify optional base config merging."""
        config = load_config("configs/snake_dqn.yaml", base_config_path="configs/base.yaml")
        assert config["game"] == "snake"
        assert config["device"] == "auto"
        assert "directories" in config
        assert config["directories"]["log_dir"] == "logs"
