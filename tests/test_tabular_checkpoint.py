"""Unit tests for Tabular Checkpoint Manager (Q-Learning and SARSA)."""

import pickle
from pathlib import Path
import numpy as np
import pytest

from src.agents.tabular.checkpoint import TabularCheckpointManager
from src.agents.tabular.q_learning import QLearningAgent
from src.agents.tabular.sarsa import SARSAAgent


@pytest.fixture
def tmp_chk_dir(tmp_path: Path) -> Path:
    """Fixture providing temporary checkpoint directory."""
    chk_dir = tmp_path / "checkpoints" / "flappy"
    chk_dir.mkdir(parents=True, exist_ok=True)
    return chk_dir


@pytest.fixture
def sample_config() -> dict:
    """Sample configuration dictionary for testing."""
    return {
        "game": "flappy",
        "agent": {"algorithm": "q_learning", "alpha": 0.1, "gamma": 0.99},
        "epsilon": {"start": 1.0, "end": 0.05, "decay_episodes": 5000},
        "seed": 42,
    }


def test_save_creates_checkpoint_file(tmp_chk_dir: Path, sample_config: dict):
    """Verify that save creates a checkpoint file in the designated directory."""
    manager = TabularCheckpointManager(checkpoint_dir=tmp_chk_dir)
    agent = QLearningAgent(num_states=75, num_actions=2, seed=42)

    saved_path = manager.save(
        agent=agent,
        episode=100,
        algorithm="q_learning",
        config=sample_config,
        seed=42,
    )

    assert saved_path.is_file()
    assert saved_path.name == "checkpoint_ep100.pkl"
    assert saved_path.parent == tmp_chk_dir


def test_saved_checkpoint_contains_required_fields(tmp_chk_dir: Path, sample_config: dict):
    """Verify envelope format and presence of all required fields."""
    manager = TabularCheckpointManager(checkpoint_dir=tmp_chk_dir)
    agent = QLearningAgent(num_states=75, num_actions=2, seed=42)
    agent.epsilon = 0.5
    agent.episode = 250

    saved_path = manager.save(
        agent=agent,
        episode=250,
        algorithm="q_learning",
        config=sample_config,
        seed=42,
        extra_info={"test_key": "test_val"},
    )

    with open(saved_path, "rb") as f:
        data = pickle.load(f)

    assert data["format_version"] == "1.0"
    assert data["algorithm"] == "q_learning"
    assert data["episode"] == 250
    assert data["epsilon"] == 0.5
    assert "agent_state" in data
    assert data["config"] == sample_config
    assert data["seed"] == 42
    assert data["extra_info"] == {"test_key": "test_val"}


def test_q_table_survives_save_load_exactly(tmp_chk_dir: Path, sample_config: dict):
    """Verify that Q-table values survive serialization and restoration bit-for-bit."""
    manager = TabularCheckpointManager(checkpoint_dir=tmp_chk_dir)
    agent = QLearningAgent(num_states=75, num_actions=2, seed=42)

    # Set arbitrary test values in Q-table
    agent.q_table[10, 0] = 3.1415926535
    agent.q_table[74, 1] = -42.0
    agent.q_table[0, 1] = 0.00001

    saved_path = manager.save(
        agent=agent,
        episode=50,
        algorithm="q_learning",
        config=sample_config,
        seed=42,
    )

    fresh_agent = QLearningAgent(num_states=75, num_actions=2)
    loaded_data = manager.load(saved_path, agent=fresh_agent)

    np.testing.assert_array_equal(fresh_agent.q_table, agent.q_table)
    assert fresh_agent.q_table is not loaded_data["agent_state"]["q_table"]


def test_episode_and_epsilon_survive_save_load(tmp_chk_dir: Path, sample_config: dict):
    """Verify episode and epsilon values survive restoration."""
    manager = TabularCheckpointManager(checkpoint_dir=tmp_chk_dir)
    agent = SARSAAgent(num_states=75, num_actions=2, seed=42)
    agent.episode = 1500
    agent.epsilon = 0.35

    saved_path = manager.save(
        agent=agent,
        episode=1500,
        algorithm="sarsa",
        config=sample_config,
        seed=42,
    )

    fresh_agent = SARSAAgent(num_states=75, num_actions=2)
    manager.load(saved_path, agent=fresh_agent)

    assert fresh_agent.episode == 1500
    assert fresh_agent.epsilon == pytest.approx(0.35)


def test_alpha_gamma_and_agent_parameters_survive(tmp_chk_dir: Path, sample_config: dict):
    """Verify hyperparameters survive restoration."""
    manager = TabularCheckpointManager(checkpoint_dir=tmp_chk_dir)
    agent = QLearningAgent(num_states=75, num_actions=2, alpha=0.25, gamma=0.95, seed=42)

    saved_path = manager.save(
        agent=agent,
        episode=10,
        algorithm="q_learning",
        config=sample_config,
        seed=42,
    )

    fresh_agent = QLearningAgent(num_states=75, num_actions=2, alpha=0.01, gamma=0.5)
    manager.load(saved_path, agent=fresh_agent)

    assert fresh_agent.alpha == pytest.approx(0.25)
    assert fresh_agent.gamma == pytest.approx(0.95)


def test_configuration_and_seed_survive(tmp_chk_dir: Path, sample_config: dict):
    """Verify config dict and seed are preserved exactly."""
    manager = TabularCheckpointManager(checkpoint_dir=tmp_chk_dir)
    agent = QLearningAgent(num_states=75, num_actions=2, seed=999)

    saved_path = manager.save(
        agent=agent,
        episode=1,
        algorithm="q_learning",
        config=sample_config,
        seed=999,
    )

    loaded_data = manager.load(saved_path)
    assert loaded_data["config"] == sample_config
    assert loaded_data["seed"] == 999


def test_rng_state_restoration_behavior(tmp_chk_dir: Path, sample_config: dict):
    """Verify RNG state can be restored or preserved based on restore_rng flag."""
    manager = TabularCheckpointManager(checkpoint_dir=tmp_chk_dir)
    agent = QLearningAgent(num_states=75, num_actions=2, seed=12345)

    # Sample random numbers to advance generator
    _ = [agent.rng.random() for _ in range(10)]

    saved_path = manager.save(
        agent=agent,
        episode=10,
        algorithm="q_learning",
        config=sample_config,
        seed=12345,
    )

    expected_next_rand = agent.rng.random()

    # Case 1: restore_rng = True
    fresh_agent_1 = QLearningAgent(num_states=75, num_actions=2, seed=99999)
    manager.load(saved_path, agent=fresh_agent_1, restore_rng=True)
    assert fresh_agent_1.rng.random() == pytest.approx(expected_next_rand)

    # Case 2: restore_rng = False
    fresh_agent_2 = QLearningAgent(num_states=75, num_actions=2, seed=99999)
    fresh_expected_rand = fresh_agent_2.rng.random()
    # Reset agent 2 to same initial seed
    fresh_agent_2 = QLearningAgent(num_states=75, num_actions=2, seed=99999)
    manager.load(saved_path, agent=fresh_agent_2, restore_rng=False)
    assert fresh_agent_2.rng.random() == pytest.approx(fresh_expected_rand)


def test_loading_into_fresh_qlearning_agent(tmp_chk_dir: Path, sample_config: dict):
    """Verify loading into fresh QLearningAgent restores behavior and Q-table."""
    manager = TabularCheckpointManager(checkpoint_dir=tmp_chk_dir)
    agent = QLearningAgent(num_states=75, num_actions=2, seed=42)
    agent.q_table[5, 1] = 10.0
    agent.q_table[5, 0] = 2.0

    saved_path = manager.save(
        agent=agent,
        episode=100,
        algorithm="q_learning",
        config=sample_config,
        seed=42,
    )

    fresh_agent = QLearningAgent(num_states=75, num_actions=2)
    manager.load(saved_path, agent=fresh_agent, expected_algorithm="q_learning")

    assert fresh_agent.select_greedy_action(5) == 1
    assert fresh_agent.q_table[5, 1] == 10.0


def test_loading_into_fresh_sarsa_agent(tmp_chk_dir: Path, sample_config: dict):
    """Verify loading into fresh SARSAAgent restores behavior and Q-table."""
    manager = TabularCheckpointManager(checkpoint_dir=tmp_chk_dir)
    agent = SARSAAgent(num_states=75, num_actions=2, seed=42)
    agent.q_table[12, 0] = 15.0
    agent.q_table[12, 1] = 1.0

    saved_path = manager.save(
        agent=agent,
        episode=100,
        algorithm="sarsa",
        config=sample_config,
        seed=42,
    )

    fresh_agent = SARSAAgent(num_states=75, num_actions=2)
    manager.load(saved_path, agent=fresh_agent, expected_algorithm="sarsa")

    assert fresh_agent.select_greedy_action(12) == 0
    assert fresh_agent.q_table[12, 0] == 15.0


def test_latest_and_list_checkpoints(tmp_chk_dir: Path, sample_config: dict):
    """Verify listing and latest checkpoint resolution by episode index."""
    manager = TabularCheckpointManager(checkpoint_dir=tmp_chk_dir)
    agent = QLearningAgent(num_states=75, num_actions=2)

    assert manager.list_checkpoints() == []
    assert manager.latest_checkpoint() is None

    # Save multiple checkpoints in non-sequential order
    manager.save(agent=agent, episode=500, algorithm="q_learning", config=sample_config, seed=42)
    manager.save(agent=agent, episode=100, algorithm="q_learning", config=sample_config, seed=42)
    manager.save(agent=agent, episode=1000, algorithm="q_learning", config=sample_config, seed=42)

    chk_list = manager.list_checkpoints()
    assert len(chk_list) == 3
    assert [p.name for p in chk_list] == [
        "checkpoint_ep100.pkl",
        "checkpoint_ep500.pkl",
        "checkpoint_ep1000.pkl",
    ]

    latest = manager.latest_checkpoint()
    assert latest is not None
    assert latest.name == "checkpoint_ep1000.pkl"


def test_exists_method(tmp_chk_dir: Path, sample_config: dict):
    """Verify exists() handles filenames and paths."""
    manager = TabularCheckpointManager(checkpoint_dir=tmp_chk_dir)
    agent = QLearningAgent(num_states=75, num_actions=2)

    assert not manager.exists("checkpoint_ep100.pkl")
    saved_path = manager.save(agent=agent, episode=100, algorithm="q_learning", config=sample_config, seed=42)

    assert manager.exists("checkpoint_ep100.pkl")
    assert manager.exists(saved_path)
    assert not manager.exists("checkpoint_ep999.pkl")


def test_malformed_checkpoint_raises_error(tmp_chk_dir: Path):
    """Verify corrupt or missing fields trigger ValueError."""
    manager = TabularCheckpointManager(checkpoint_dir=tmp_chk_dir)

    corrupt_file = tmp_chk_dir / "corrupt.pkl"
    with open(corrupt_file, "wb") as f:
        pickle.dump({"invalid": "data"}, f)

    with pytest.raises(ValueError, match="missing required field"):
        manager.load(corrupt_file)

    bad_version_file = tmp_chk_dir / "bad_version.pkl"
    with open(bad_version_file, "wb") as f:
        pickle.dump(
            {
                "format_version": "99.0",
                "algorithm": "q_learning",
                "episode": 1,
                "epsilon": 1.0,
                "agent_state": {"q_table": np.zeros((75, 2))},
            },
            f,
        )

    with pytest.raises(ValueError, match="Incompatible checkpoint format version"):
        manager.load(bad_version_file)


def test_incompatible_q_table_shape_rejected(tmp_chk_dir: Path, sample_config: dict):
    """Verify Q-table shape mismatch between checkpoint and agent raises ValueError."""
    manager = TabularCheckpointManager(checkpoint_dir=tmp_chk_dir)
    small_agent = QLearningAgent(num_states=10, num_actions=2)

    saved_path = manager.save(
        agent=small_agent,
        episode=1,
        algorithm="q_learning",
        config=sample_config,
        seed=42,
    )

    standard_agent = QLearningAgent(num_states=75, num_actions=2)
    with pytest.raises(ValueError, match="Incompatible Q-table shape"):
        manager.load(saved_path, agent=standard_agent)


def test_algorithm_mismatch_validation(tmp_chk_dir: Path, sample_config: dict):
    """Verify expected_algorithm mismatch triggers ValueError."""
    manager = TabularCheckpointManager(checkpoint_dir=tmp_chk_dir)
    agent = QLearningAgent(num_states=75, num_actions=2)

    saved_path = manager.save(
        agent=agent,
        episode=1,
        algorithm="q_learning",
        config=sample_config,
        seed=42,
    )

    with pytest.raises(ValueError, match="Algorithm mismatch"):
        manager.load(saved_path, expected_algorithm="sarsa")


def test_atomic_write_no_tmp_leftover(tmp_chk_dir: Path, sample_config: dict):
    """Verify atomic write leaves only the target file and removes temporary file."""
    manager = TabularCheckpointManager(checkpoint_dir=tmp_chk_dir)
    agent = QLearningAgent(num_states=75, num_actions=2)

    saved_path = manager.save(
        agent=agent,
        episode=50,
        algorithm="q_learning",
        config=sample_config,
        seed=42,
    )

    assert saved_path.is_file()
    tmp_files = list(tmp_chk_dir.glob("*.tmp"))
    assert len(tmp_files) == 0
