"""Unit and integration tests for formal Flappy Bird tabular RL evaluation."""

import csv
from pathlib import Path
import pickle
import subprocess
import sys
import numpy as np
import pytest

from evaluate_flappy import (
    FLAPPY_EVALUATION_COLUMNS,
    create_agent,
    export_clean_model,
    load_model_file,
    run_evaluation,
)
from src.agents.tabular.checkpoint import TabularCheckpointManager
from src.agents.tabular.q_learning import QLearningAgent
from src.agents.tabular.sarsa import SARSAAgent


@pytest.fixture
def temp_eval_dirs(tmp_path: Path):
    """Fixture providing temporary directories for checkpoints, models, and evaluation outputs."""
    chk_dir = tmp_path / "checkpoints"
    mdl_dir = tmp_path / "models"
    eval_dir = tmp_path / "evaluations"
    chk_dir.mkdir(parents=True, exist_ok=True)
    mdl_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    return {
        "chk_dir": chk_dir,
        "mdl_dir": mdl_dir,
        "eval_dir": eval_dir,
        "tmp_path": tmp_path,
    }


class TestEvaluateFlappyPipeline:
    """Test suite for evaluate_flappy.py execution, CLI, model loading, immutability, and export."""

    def test_cli_help(self):
        """1. Verify CLI --help option executes with return code 0 and displays all required options."""
        cmd = [sys.executable, "evaluate_flappy.py", "--help"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0
        assert "Evaluate Trained Flappy Bird Tabular RL Agent" in res.stdout
        assert "--algorithm" in res.stdout
        assert "--model" in res.stdout
        assert "--episodes" in res.stdout
        assert "--seed" in res.stdout
        assert "--output-dir" in res.stdout
        assert "--export-model" in res.stdout

    def test_checkpoint_and_clean_model_loading(self, temp_eval_dirs):
        """2. Verify loading from both full checkpoint envelopes and clean exported models."""
        chk_dir = temp_eval_dirs["chk_dir"]
        chk_mgr = TabularCheckpointManager(checkpoint_dir=chk_dir)

        # Create dummy trained agent with distinct non-zero Q-values
        agent = QLearningAgent(num_states=75, num_actions=2, seed=42)
        agent.q_table[10, 0] = 42.5
        agent.q_table[20, 1] = -12.3

        # Save training checkpoint envelope
        chk_path = chk_mgr.save(agent=agent, episode=5000, algorithm="q_learning", filename="test_chk.pkl")
        assert chk_path.is_file()

        # Load into fresh agent
        fresh_agent = QLearningAgent(num_states=75, num_actions=2, seed=99)
        load_model_file(chk_path, agent=fresh_agent, expected_algorithm="q_learning")
        np.testing.assert_array_equal(agent.q_table, fresh_agent.q_table)

        # Export clean model artifact
        export_path = temp_eval_dirs["mdl_dir"] / "test_clean_model.pkl"
        export_clean_model(agent=agent, algorithm="q_learning", export_path=export_path)
        assert export_path.is_file()

        # Load clean model artifact into another fresh agent
        clean_load_agent = QLearningAgent(num_states=75, num_actions=2, seed=123)
        load_model_file(export_path, agent=clean_load_agent, expected_algorithm="q_learning")
        np.testing.assert_array_equal(agent.q_table, clean_load_agent.q_table)

    def test_algorithm_mismatch_detection(self, temp_eval_dirs):
        """3. Verify algorithm mismatch raises ValueError when loading checkpoint."""
        chk_dir = temp_eval_dirs["chk_dir"]
        chk_mgr = TabularCheckpointManager(checkpoint_dir=chk_dir)

        sarsa_agent = SARSAAgent(num_states=75, num_actions=2, seed=42)
        sarsa_chk = chk_mgr.save(agent=sarsa_agent, episode=100, algorithm="sarsa", filename="sarsa_chk.pkl")

        ql_agent = QLearningAgent(num_states=75, num_actions=2, seed=42)
        with pytest.raises(ValueError, match="Algorithm mismatch"):
            load_model_file(sarsa_chk, agent=ql_agent, expected_algorithm="q_learning")

    def test_short_q_learning_evaluation_and_csv_generation(self, temp_eval_dirs):
        """4. Verify short 3-episode Q-Learning evaluation run produces valid CSV and returns correct stats."""
        chk_dir = temp_eval_dirs["chk_dir"]
        eval_dir = temp_eval_dirs["eval_dir"]
        chk_mgr = TabularCheckpointManager(checkpoint_dir=chk_dir)

        agent = QLearningAgent(num_states=75, num_actions=2, seed=42)
        agent.q_table[5, 0] = 10.0
        chk_path = chk_mgr.save(agent=agent, episode=1000, algorithm="q_learning", filename="ql_chk.pkl")

        stats = run_evaluation(
            algorithm="q_learning",
            model_path=str(chk_path),
            episodes=3,
            seed=42,
            output_dir=str(eval_dir),
            export_model=False,
        )

        assert stats["algorithm"] == "q_learning"
        assert stats["episodes"] == 3
        assert stats["epsilon"] == 0.0
        assert stats["q_table_integrity_passed"] is True
        assert stats["min_length"] >= 1

        csv_path = eval_dir / "q_learning.csv"
        assert csv_path.is_file()

        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 3
        for col in FLAPPY_EVALUATION_COLUMNS:
            assert col in rows[0]
        assert [r["episode"] for r in rows] == ["1", "2", "3"]
        assert all(r["seed"] == "42" for r in rows)

    def test_short_sarsa_evaluation_and_export_model(self, temp_eval_dirs):
        """5. Verify short 3-episode SARSA evaluation and clean model export."""
        chk_dir = temp_eval_dirs["chk_dir"]
        eval_dir = temp_eval_dirs["eval_dir"]
        mdl_dir = temp_eval_dirs["mdl_dir"]
        chk_mgr = TabularCheckpointManager(checkpoint_dir=chk_dir)

        agent = SARSAAgent(num_states=75, num_actions=2, seed=100)
        agent.q_table[15, 1] = 5.0
        chk_path = chk_mgr.save(agent=agent, episode=5000, algorithm="sarsa", filename="sarsa_chk.pkl")

        export_target = mdl_dir / "flappy_sarsa_table.pkl"

        stats = run_evaluation(
            algorithm="sarsa",
            model_path=str(chk_path),
            episodes=3,
            seed=100,
            output_dir=str(eval_dir),
            export_model=True,
            export_path=str(export_target),
        )

        assert stats["algorithm"] == "sarsa"
        assert stats["episodes"] == 3
        assert stats["q_table_integrity_passed"] is True
        assert export_target.is_file()

        # Verify exported clean model contents
        with open(export_target, "rb") as f:
            exported_data = pickle.load(f)

        assert exported_data["model_type"] == "tabular_q_table"
        assert exported_data["algorithm"] == "sarsa"
        assert exported_data["num_states"] == 75
        assert exported_data["num_actions"] == 2
        assert "rng_state" not in exported_data  # Clean artifact: no training RNG
        assert "epsilon" not in exported_data    # Clean artifact: no training exploration

        np.testing.assert_array_equal(exported_data["q_table"], agent.q_table)

    def test_eval_mode_enforces_zero_exploration(self):
        """6. Verify select_action(eval_mode=True) deterministically selects greedy argmax."""
        agent = QLearningAgent(num_states=75, num_actions=2, seed=42)
        agent.epsilon = 1.0  # Training epsilon is set to full exploration

        # State 0: Q(0, 0)=1.0, Q(0, 1)=5.0 -> greedy is 1
        agent.q_table[0, 0] = 1.0
        agent.q_table[0, 1] = 5.0

        for _ in range(20):
            action = agent.select_action(0, eval_mode=True)
            assert action == 1

        # State 1: Q(1, 0)=10.0, Q(1, 1)=2.0 -> greedy is 0
        agent.q_table[1, 0] = 10.0
        agent.q_table[1, 1] = 2.0

        for _ in range(20):
            action = agent.select_action(1, eval_mode=True)
            assert action == 0

    def test_q_table_integrity_failure_detection(self, temp_eval_dirs, monkeypatch):
        """7. Verify evaluator detects and fails loudly if Q-table is modified during evaluation."""
        chk_dir = temp_eval_dirs["chk_dir"]
        chk_mgr = TabularCheckpointManager(checkpoint_dir=chk_dir)
        agent = QLearningAgent(num_states=75, num_actions=2, seed=42)
        chk_path = chk_mgr.save(agent=agent, episode=10, algorithm="q_learning", filename="ql_chk.pkl")

        # Monkeypatch select_action to illegally mutate Q-table during evaluation
        orig_select = QLearningAgent.select_action

        def malicious_select(self, state, eval_mode=False):
            self.q_table[0, 0] += 999.0
            return orig_select(self, state, eval_mode)

        monkeypatch.setattr(QLearningAgent, "select_action", malicious_select)

        with pytest.raises(RuntimeError, match="CRITICAL INTEGRITY FAILURE: Q-table was modified"):
            run_evaluation(
                algorithm="q_learning",
                model_path=str(chk_path),
                episodes=1,
                output_dir=str(temp_eval_dirs["eval_dir"]),
            )
