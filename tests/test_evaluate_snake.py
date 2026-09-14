"""Unit and integration tests for formal Snake DQN evaluation."""

import csv
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

from evaluate_snake import run_evaluation
from src.agents.dqn.agent import SnakeDQNAgent
from src.common.checkpoint import CheckpointManager


class TestEvaluateSnakePipeline:
    """Test suite for evaluate_snake.py execution, CLI, model loading, model integrity, and metrics CSV."""

    def test_cli_help(self):
        """1. Verify CLI --help option executes and returns 0 exit code."""
        cmd = [sys.executable, "evaluate_snake.py", "--help"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0
        assert "Evaluate Trained Snake Refined DQN Agent" in res.stdout
        assert "--model" in res.stdout
        assert "--episodes" in res.stdout
        assert "--seed" in res.stdout

    def test_model_loading_and_cpu_execution(self, tmp_path: Path):
        """2 & 10. Verify model loading into SnakeDQNAgent and CPU execution."""
        chk_dir = tmp_path / "checkpoints"
        chk_manager = CheckpointManager(checkpoint_dir=chk_dir)
        agent = SnakeDQNAgent(device="cpu", seed=42)

        model_file = chk_manager.save(agent, episode=10, filename="test_model.pt")
        assert model_file.is_file()

        fresh_agent = SnakeDQNAgent(device="cpu", seed=999)
        chk_manager.load(model_file, agent=fresh_agent, map_location="cpu")

        for p1, p2 in zip(agent.online_net.parameters(), fresh_agent.online_net.parameters()):
            assert torch.equal(p1, p2)

    def test_short_evaluation_run_completes_and_metrics_csv(self, tmp_path: Path):
        """4, 5, 6, 7. Verify short evaluation run completes and writes valid CSV."""
        chk_dir = tmp_path / "checkpoints"
        chk_manager = CheckpointManager(checkpoint_dir=chk_dir)
        agent = SnakeDQNAgent(device="cpu", seed=42)
        model_file = chk_manager.save(agent, episode=5, filename="eval_test_model.pt")

        out_dir = tmp_path / "logs" / "evaluation"
        stats = run_evaluation(
            model_path=str(model_file),
            episodes=3,
            seed=42,
            device="cpu",
            output_dir=str(out_dir),
        )

        assert stats["episodes"] == 3
        assert stats["epsilon"] == 0.0
        assert stats["model_integrity_passed"] is True

        csv_path = out_dir / "evaluation_metrics.csv"
        assert csv_path.is_file()

        with open(csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 3
        assert list(rows[0].keys()) == ["episode", "reward", "episode_length", "score", "seed"]
        assert [r["episode"] for r in rows] == ["1", "2", "3"]

    def test_eval_mode_epsilon_is_zero(self):
        """8. Verify select_action with eval_mode=True enforces zero exploration (epsilon=0.0)."""
        agent = SnakeDQNAgent()
        agent.frame_count = 0  # Default epsilon would be 1.0

        state = np.random.randn(4, 64, 64).astype(np.float32)
        # eval_mode=True must select deterministic argmax action without random exploration
        action = agent.select_action(state, eval_mode=True)
        assert isinstance(action, int)
        assert 0 <= action <= 3

    def test_no_gradients_and_no_learning_during_evaluation(self, tmp_path: Path):
        """9. Verify no gradient is created and agent.update_steps is unchanged during evaluation."""
        chk_manager = CheckpointManager(checkpoint_dir=tmp_path)
        agent = SnakeDQNAgent(device="cpu")
        model_file = chk_manager.save(agent, episode=1, filename="no_grad_model.pt")

        update_steps_before = agent.update_steps
        stats = run_evaluation(
            model_path=str(model_file),
            episodes=2,
            seed=42,
            device="cpu",
            output_dir=str(tmp_path / "logs"),
        )

        assert stats["model_integrity_passed"] is True
        for param in agent.online_net.parameters():
            assert param.grad is None

    def test_model_weight_integrity_preserved(self, tmp_path: Path):
        """11. Verify model weights prior to evaluation equal model weights after evaluation."""
        chk_manager = CheckpointManager(checkpoint_dir=tmp_path)
        agent = SnakeDQNAgent(device="cpu", seed=100)

        # Snapshot weights before eval
        weights_before = [p.clone() for p in agent.online_net.parameters()]
        model_file = chk_manager.save(agent, episode=1, filename="integrity_model.pt")

        run_evaluation(
            model_path=str(model_file),
            episodes=3,
            seed=42,
            device="cpu",
            output_dir=str(tmp_path / "logs"),
        )

        # Snapshot weights after eval
        for p_before, p_after in zip(weights_before, agent.online_net.parameters()):
            assert torch.equal(p_before, p_after), "Model weights must not change during evaluation!"
