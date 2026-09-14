"""Unit and integration tests for Snake DQN training loop."""

import csv
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest

from train_snake import run_training, setup_seed
from src.environments.snake_env import SnakeEnv
from src.preprocessing.frame_stack import FrameStack
from src.preprocessing.snake_preprocess import SnakePreprocessor


class TestTrainSnakePipeline:
    """Test suite for Snake DQN training pipeline execution, CLI, pipeline shapes, metrics, and checkpoints."""

    def test_cli_help(self):
        """1. Verify CLI --help option executes and returns 0 exit code."""
        cmd = [sys.executable, "train_snake.py", "--help"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0
        assert "Train Snake Refined DQN Agent" in res.stdout
        assert "--config" in res.stdout
        assert "--episodes" in res.stdout
        assert "--resume" in res.stdout

    def test_short_training_run_completes(self, tmp_path: Path):
        """2 & 10. Verify a short CPU training run completes without errors."""
        res = run_training(
            config_path="configs/snake_dqn.yaml",
            episodes=2,
            seed=42,
            device="cpu",
            log_dir=str(tmp_path / "logs"),
            checkpoint_dir=str(tmp_path / "checkpoints"),
            model_dir=str(tmp_path / "models"),
        )

        assert isinstance(res, dict)
        assert res["total_episodes_run"] == 2
        assert res["final_frame_count"] > 0
        assert res["end_episode"] == 2

    def test_observation_pipeline_stacked_shape(self):
        """3. Verify environment -> preprocessor -> frame stack produces shape (4, 64, 64)."""
        env = SnakeEnv(obs_type="rgb", random_spawn=False)
        preprocessor = SnakePreprocessor(target_shape=(64, 64), dtype=np.float32)
        frame_stack = FrameStack(stack_size=4, frame_shape=(64, 64), dtype=np.float32)

        raw_obs, _ = env.reset(seed=42)
        proc_frame = preprocessor.transform(raw_obs)
        stacked = frame_stack.reset(proc_frame)

        assert stacked.shape == (4, 64, 64)
        assert stacked.dtype == np.float32

        # Step next frame
        next_raw, _, _, _, _ = env.step(0)
        next_proc = preprocessor.transform(next_raw)
        next_stacked = frame_stack.step(next_proc)

        assert next_stacked.shape == (4, 64, 64)

    def test_metrics_csv_creation_and_row_content(self, tmp_path: Path):
        """4, 5, 9. Verify metrics CSV creation, episode row writing, and seed recording."""
        log_dir = tmp_path / "logs"
        run_training(
            episodes=3,
            seed=123,
            device="cpu",
            log_dir=str(log_dir),
            checkpoint_dir=str(tmp_path / "checkpoints"),
            model_dir=str(tmp_path / "models"),
        )

        csv_path = log_dir / "training_metrics.csv"
        assert csv_path.is_file()

        with open(csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 3
        assert rows[0]["episode"] == "1"
        assert rows[0]["seed"] == "123"
        assert "reward" in rows[0]
        assert "score" in rows[0]
        assert "epsilon" in rows[0]

    def test_checkpoint_creation_and_resume_flow(self, tmp_path: Path):
        """6 & 7. Verify checkpoint creation during short run and resuming from checkpoint."""
        chk_dir = tmp_path / "checkpoints"
        log_dir = tmp_path / "logs"

        # Session 1: Run 2 episodes with checkpoint saving after every episode
        run_training(
            episodes=2,
            seed=42,
            device="cpu",
            log_dir=str(log_dir),
            checkpoint_dir=str(chk_dir),
            model_dir=str(tmp_path / "models"),
            checkpoint_interval_override=1,
        )

        chk_file = chk_dir / "checkpoint_ep2.pt"
        assert chk_file.is_file()

        # Session 2: Resume from checkpoint_ep2.pt and run 2 more episodes (episodes 3 and 4)
        res_resume = run_training(
            episodes=2,
            seed=42,
            resume_path=str(chk_file),
            device="cpu",
            log_dir=str(log_dir),
            checkpoint_dir=str(chk_dir),
            model_dir=str(tmp_path / "models"),
            checkpoint_interval_override=1,
        )

        assert res_resume["start_episode"] == 3
        assert res_resume["end_episode"] == 4

        # Check metrics CSV has 4 total rows
        csv_path = log_dir / "training_metrics.csv"
        with open(csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 4
        assert [r["episode"] for r in rows] == ["1", "2", "3", "4"]

    def test_best_model_file_production(self, tmp_path: Path):
        """8. Verify best-model file is created."""
        mdl_dir = tmp_path / "models"
        res = run_training(
            episodes=2,
            seed=42,
            device="cpu",
            log_dir=str(tmp_path / "logs"),
            checkpoint_dir=str(tmp_path / "checkpoints"),
            model_dir=str(mdl_dir),
        )

        best_path = Path(res["best_model_path"])
        assert best_path.is_file()
        assert best_path.name == "snake_dqn_best.pt"
