"""Unit tests for CheckpointManager implementation."""

import random
from pathlib import Path
import numpy as np
import pytest
import torch

from src.agents.dqn.agent import SnakeDQNAgent
from src.common.checkpoint import CheckpointManager


class TestCheckpointManager:
    """Test suite for CheckpointManager serialization, restoration, best export, and RNG state handling."""

    def test_checkpoint_manager_initialization(self, tmp_path: Path):
        """1 & 2. Verify initialization and directory creation."""
        chk_dir = tmp_path / "checkpoints" / "snake"
        assert not chk_dir.exists()

        manager = CheckpointManager(checkpoint_dir=chk_dir, format_version="1.0")
        assert manager.checkpoint_dir == chk_dir
        assert manager.format_version == "1.0"
        assert chk_dir.is_dir()

    def test_save_and_file_existence(self, tmp_path: Path):
        """3 & 4. Verify saving a checkpoint produces an existing file."""
        manager = CheckpointManager(checkpoint_dir=tmp_path / "checkpoints")
        agent = SnakeDQNAgent()

        saved_path = manager.save(agent=agent, episode=50, filename="checkpoint_ep50.pt")
        assert saved_path.is_file()
        assert manager.exists("checkpoint_ep50.pt")

    def test_load_into_fresh_agent_and_weight_restoration(self, tmp_path: Path):
        """5, 6, 7, 8. Verify loading restores online weights, target weights, and optimizer state."""
        manager = CheckpointManager(checkpoint_dir=tmp_path / "checkpoints")
        agent_orig = SnakeDQNAgent(seed=42)

        # Mutate online network weights
        with torch.no_grad():
            for p in agent_orig.online_net.parameters():
                p.add_(0.5)
        agent_orig.update_target_network()

        # Perform a dummy training step to create non-empty optimizer state
        state = np.random.randn(4, 64, 64).astype(np.float32)
        for _ in range(35):
            agent_orig.store_transition(state, 0, 1.0, state, False)
        agent_orig.learn()

        saved_path = manager.save(agent=agent_orig, episode=10)

        # Fresh agent with different initial seed
        fresh_agent = SnakeDQNAgent(seed=999)

        # Verify weights differ prior to loading
        assert any(
            not torch.equal(p1, p2)
            for p1, p2 in zip(fresh_agent.online_net.parameters(), agent_orig.online_net.parameters())
        )

        data = manager.load(saved_path, agent=fresh_agent)
        assert isinstance(data, dict)

        # 6. Verify online weights restored exactly
        for p1, p2 in zip(fresh_agent.online_net.parameters(), agent_orig.online_net.parameters()):
            assert torch.equal(p1, p2)

        # 7. Verify target weights restored exactly
        for p1, p2 in zip(fresh_agent.target_net.parameters(), agent_orig.target_net.parameters()):
            assert torch.equal(p1, p2)

        # 8. Verify optimizer state restored
        assert len(fresh_agent.optimizer.state_dict()["state"]) > 0

    def test_counters_and_epsilon_restoration(self, tmp_path: Path):
        """9, 10, 11, 12, 13. Verify frame count, update steps, episode, epsilon, and config restoration."""
        manager = CheckpointManager(checkpoint_dir=tmp_path / "checkpoints")
        agent_orig = SnakeDQNAgent()
        agent_orig.frame_count = 75000
        agent_orig.update_steps = 150

        saved_path = manager.save(agent=agent_orig, episode=42)

        fresh_agent = SnakeDQNAgent()
        data = manager.load(saved_path, agent=fresh_agent)

        # 9. Frame count
        assert fresh_agent.frame_count == 75000
        # 10. Update steps
        assert fresh_agent.update_steps == 150
        # 11. Episode value
        assert data["episode"] == 42
        # 12. Epsilon state
        assert pytest.approx(fresh_agent.get_epsilon(), abs=1e-4) == agent_orig.get_epsilon()
        # 13. Config stored
        assert data["config"] == agent_orig.config

    def test_cpu_loading_explicit(self, tmp_path: Path):
        """14. Verify loading onto CPU explicitly works cleanly."""
        manager = CheckpointManager(checkpoint_dir=tmp_path / "checkpoints")
        agent = SnakeDQNAgent()
        saved_path = manager.save(agent=agent, episode=1)

        fresh_agent = SnakeDQNAgent(device="cpu")
        data = manager.load(saved_path, agent=fresh_agent, map_location="cpu")
        assert fresh_agent.device.type == "cpu"
        assert isinstance(data, dict)

    def test_saved_and_loaded_agent_equivalence(self, tmp_path: Path):
        """15. Verify a saved and loaded agent produces identical action selection outputs."""
        manager = CheckpointManager(checkpoint_dir=tmp_path / "checkpoints")
        agent_orig = SnakeDQNAgent(seed=123)
        agent_orig.frame_count = 50000

        saved_path = manager.save(agent=agent_orig, episode=10)

        fresh_agent = SnakeDQNAgent(seed=999)
        manager.load(saved_path, agent=fresh_agent)

        state = np.random.randn(4, 64, 64).astype(np.float32)
        act_orig = agent_orig.select_action(state, eval_mode=True)
        act_fresh = fresh_agent.select_action(state, eval_mode=True)
        assert act_orig == act_fresh

    def test_best_model_export(self, tmp_path: Path):
        """16. Verify best-model export can be saved separately from periodic checkpoints."""
        chk_dir = tmp_path / "checkpoints"
        best_file = tmp_path / "models" / "snake_dqn_best.pt"
        manager = CheckpointManager(checkpoint_dir=chk_dir)
        agent = SnakeDQNAgent()

        periodic_path = manager.save(
            agent=agent,
            episode=100,
            filename="checkpoint_ep100.pt",
            is_best=True,
            best_filename=best_file,
        )

        assert periodic_path.is_file()
        assert best_file.is_file()
        assert periodic_path != best_file

    def test_checkpoint_format_version_field(self, tmp_path: Path):
        """17. Verify checkpoint format_version field exists in output data."""
        manager = CheckpointManager(checkpoint_dir=tmp_path / "checkpoints", format_version="1.0")
        agent = SnakeDQNAgent()
        saved_path = manager.save(agent=agent, episode=5)

        data = manager.load(saved_path)
        assert "format_version" in data
        assert data["format_version"] == "1.0"

    def test_unrelated_files_not_overwritten(self, tmp_path: Path):
        """18. Verify saving does not overwrite unrelated files in directory."""
        chk_dir = tmp_path / "checkpoints"
        chk_dir.mkdir(parents=True, exist_ok=True)
        unrelated_file = chk_dir / "unrelated.txt"
        unrelated_file.write_text("important data", encoding="utf-8")

        manager = CheckpointManager(checkpoint_dir=chk_dir)
        agent = SnakeDQNAgent()
        manager.save(agent=agent, episode=1, filename="checkpoint_ep1.pt")

        assert unrelated_file.is_file()
        assert unrelated_file.read_text(encoding="utf-8") == "important data"

    def test_rng_state_restoration(self, tmp_path: Path):
        """19. Verify Python, NumPy, and PyTorch RNG state restoration."""
        manager = CheckpointManager(checkpoint_dir=tmp_path / "checkpoints")
        agent = SnakeDQNAgent()

        # Seed initial RNGs
        random.seed(1234)
        np.random.seed(5678)
        torch.manual_seed(9012)

        # Save checkpoint to record RNG state
        saved_path = manager.save(agent=agent, episode=1)

        # Record next draw after save
        r_val_1 = random.random()
        np_val_1 = np.random.rand()
        torch_val_1 = torch.rand(1).item()

        # Advance RNGs further
        random.random()
        np.random.rand()
        torch.rand(1)

        # Restore RNG state from saved checkpoint
        manager.load(saved_path, restore_rng=True)

        # Verify next draw matches r_val_1, np_val_1, torch_val_1
        r_val_2 = random.random()
        np_val_2 = np.random.rand()
        torch_val_2 = torch.rand(1).item()

        assert r_val_1 == r_val_2
        assert np_val_1 == np_val_2
        assert torch_val_1 == torch_val_2

    def test_file_not_found_and_invalid_format_raises(self, tmp_path: Path):
        """Verify error handling for missing or malformed checkpoint files."""
        manager = CheckpointManager(checkpoint_dir=tmp_path / "checkpoints")

        with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
            manager.load(tmp_path / "non_existent.pt")

        bad_file = tmp_path / "bad.pt"
        torch.save({"unrelated": 123}, bad_file)
        with pytest.raises(ValueError, match="Invalid checkpoint format"):
            manager.load(bad_file)
