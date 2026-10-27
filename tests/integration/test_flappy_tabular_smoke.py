"""Integration Smoke Tests for Flappy Tabular Training Runner (Q-Learning and SARSA)."""

import csv
from pathlib import Path
import numpy as np
import pytest

from train_flappy_tabular import (
    TABULAR_METRICS_COLUMNS,
    run_tabular_training,
)
from src.agents.tabular.checkpoint import TabularCheckpointManager


@pytest.fixture
def temp_training_dirs(tmp_path: Path):
    """Fixture providing temporary directories for logs, checkpoints, and models."""
    log_dir = tmp_path / "logs"
    chk_dir = tmp_path / "checkpoints"
    mdl_dir = tmp_path / "models"
    log_dir.mkdir(parents=True, exist_ok=True)
    chk_dir.mkdir(parents=True, exist_ok=True)
    mdl_dir.mkdir(parents=True, exist_ok=True)
    return {
        "log_dir": str(log_dir),
        "checkpoint_dir": str(chk_dir),
        "model_dir": str(mdl_dir),
        "tmp_path": tmp_path,
    }


def test_q_learning_short_training_smoke(temp_training_dirs):
    """Verify Q-Learning executes short 3-episode run, logs metrics, and updates Q-table."""
    res = run_tabular_training(
        config_path="configs/flappy_tabular.yaml",
        algorithm="q_learning",
        episodes=3,
        seed=42,
        log_dir=temp_training_dirs["log_dir"],
        checkpoint_dir=temp_training_dirs["checkpoint_dir"],
        model_dir=temp_training_dirs["model_dir"],
        checkpoint_interval=2,
    )

    assert res["algorithm"] == "q_learning"
    assert res["total_episodes_run"] == 3
    assert res["start_episode"] == 1
    assert res["end_episode"] == 3
    assert res["final_epsilon"] < 1.0  # Epsilon decayed from 1.0
    assert res["non_zero_entries"] > 0  # Q-table updated from all zeros

    # Check metrics CSV
    csv_path = Path(res["metrics_csv_path"])
    assert csv_path.is_file()

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 3
    for col in TABULAR_METRICS_COLUMNS:
        assert col in rows[0]

    # Verify periodic and best checkpoints created
    chk_mgr = TabularCheckpointManager(checkpoint_dir=temp_training_dirs["checkpoint_dir"])
    assert chk_mgr.exists("checkpoint_ep2.pkl")
    assert chk_mgr.exists("flappy_q_learning_best.pkl")


def test_sarsa_short_training_smoke(temp_training_dirs):
    """Verify SARSA executes short 3-episode run, logs metrics, and updates Q-table."""
    res = run_tabular_training(
        config_path="configs/flappy_tabular.yaml",
        algorithm="sarsa",
        episodes=3,
        seed=100,
        log_dir=temp_training_dirs["log_dir"],
        checkpoint_dir=temp_training_dirs["checkpoint_dir"],
        model_dir=temp_training_dirs["model_dir"],
        checkpoint_interval=2,
    )

    assert res["algorithm"] == "sarsa"
    assert res["total_episodes_run"] == 3
    assert res["start_episode"] == 1
    assert res["end_episode"] == 3
    assert res["final_epsilon"] < 1.0
    assert res["non_zero_entries"] > 0

    # Check metrics CSV
    csv_path = Path(res["metrics_csv_path"])
    assert csv_path.is_file()

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 3
    for col in TABULAR_METRICS_COLUMNS:
        assert col in rows[0]

    # Verify checkpoint created
    chk_mgr = TabularCheckpointManager(checkpoint_dir=temp_training_dirs["checkpoint_dir"])
    assert chk_mgr.exists("checkpoint_ep2.pkl")
    assert chk_mgr.exists("flappy_sarsa_best.pkl")


def test_invalid_algorithm_rejected(temp_training_dirs):
    """Verify specifying invalid algorithm raises ValueError."""
    with pytest.raises(ValueError, match="Invalid algorithm"):
        run_tabular_training(
            config_path="configs/flappy_tabular.yaml",
            algorithm="dqn_tabular",
            episodes=1,
            log_dir=temp_training_dirs["log_dir"],
            checkpoint_dir=temp_training_dirs["checkpoint_dir"],
        )


def test_resume_training_q_learning(temp_training_dirs):
    """Verify resuming Q-Learning training from checkpoint continues episode count and Q-table."""
    # Run initial 2 episodes
    res1 = run_tabular_training(
        config_path="configs/flappy_tabular.yaml",
        algorithm="q_learning",
        episodes=2,
        seed=42,
        log_dir=temp_training_dirs["log_dir"],
        checkpoint_dir=temp_training_dirs["checkpoint_dir"],
        checkpoint_interval=2,
    )
    checkpoint_file = Path(temp_training_dirs["checkpoint_dir"]) / "checkpoint_ep2.pkl"
    assert checkpoint_file.is_file()

    # Resume for 2 more episodes
    res2 = run_tabular_training(
        config_path="configs/flappy_tabular.yaml",
        resume_path=str(checkpoint_file),
        episodes=2,
        seed=42,
        log_dir=temp_training_dirs["log_dir"],
        checkpoint_dir=temp_training_dirs["checkpoint_dir"],
        checkpoint_interval=2,
    )

    assert res2["start_episode"] == 3
    assert res2["end_episode"] == 4
    assert res2["total_episodes_run"] == 2
    assert res2["final_epsilon"] < res1["final_epsilon"]


def test_resume_training_sarsa(temp_training_dirs):
    """Verify resuming SARSA training from checkpoint continues episode count and Q-table."""
    res1 = run_tabular_training(
        config_path="configs/flappy_tabular.yaml",
        algorithm="sarsa",
        episodes=2,
        seed=100,
        log_dir=temp_training_dirs["log_dir"],
        checkpoint_dir=temp_training_dirs["checkpoint_dir"],
        checkpoint_interval=2,
    )
    checkpoint_file = Path(temp_training_dirs["checkpoint_dir"]) / "checkpoint_ep2.pkl"
    assert checkpoint_file.is_file()

    res2 = run_tabular_training(
        config_path="configs/flappy_tabular.yaml",
        resume_path=str(checkpoint_file),
        episodes=2,
        seed=100,
        log_dir=temp_training_dirs["log_dir"],
        checkpoint_dir=temp_training_dirs["checkpoint_dir"],
        checkpoint_interval=2,
    )

    assert res2["start_episode"] == 3
    assert res2["end_episode"] == 4
    assert res2["total_episodes_run"] == 2
    assert res2["final_epsilon"] < res1["final_epsilon"]


def test_algorithm_mismatch_on_resume_rejected(temp_training_dirs):
    """Verify attempting to resume a Q-Learning checkpoint with SARSA raises ValueError."""
    # Save a Q-learning checkpoint
    run_tabular_training(
        config_path="configs/flappy_tabular.yaml",
        algorithm="q_learning",
        episodes=2,
        seed=42,
        log_dir=temp_training_dirs["log_dir"],
        checkpoint_dir=temp_training_dirs["checkpoint_dir"],
        checkpoint_interval=2,
    )
    checkpoint_file = Path(temp_training_dirs["checkpoint_dir"]) / "checkpoint_ep2.pkl"

    # Attempt to resume with sarsa
    with pytest.raises(ValueError, match="Algorithm mismatch"):
        run_tabular_training(
            config_path="configs/flappy_tabular.yaml",
            algorithm="sarsa",
            resume_path=str(checkpoint_file),
            episodes=1,
            log_dir=temp_training_dirs["log_dir"],
            checkpoint_dir=temp_training_dirs["checkpoint_dir"],
        )


def test_reproducibility_deterministic_seeding(temp_training_dirs):
    """Verify identical seeds produce identical Q-tables and trajectories."""
    dir_a = temp_training_dirs["tmp_path"] / "run_a"
    dir_b = temp_training_dirs["tmp_path"] / "run_b"

    res_a = run_tabular_training(
        config_path="configs/flappy_tabular.yaml",
        algorithm="q_learning",
        episodes=3,
        seed=777,
        log_dir=str(dir_a / "logs"),
        checkpoint_dir=str(dir_a / "checkpoints"),
        checkpoint_interval=3,
    )

    res_b = run_tabular_training(
        config_path="configs/flappy_tabular.yaml",
        algorithm="q_learning",
        episodes=3,
        seed=777,
        log_dir=str(dir_b / "logs"),
        checkpoint_dir=str(dir_b / "checkpoints"),
        checkpoint_interval=3,
    )

    np.testing.assert_array_equal(res_a["q_table"], res_b["q_table"])
    assert res_a["final_epsilon"] == pytest.approx(res_b["final_epsilon"])
