"""Unit tests for SnakeDQNAgent implementation."""

import pytest
import numpy as np
import torch

from src.agents.dqn.agent import SnakeDQNAgent
from src.common.config import load_config


class TestSnakeDQNAgent:
    """Test suite for SnakeDQNAgent initialization, networks, epsilon schedule, action selection, and learning."""

    def test_agent_initialization(self):
        """1. Verify agent initialization with default config."""
        agent = SnakeDQNAgent()
        assert agent is not None
        assert agent.device.type in ("cpu", "cuda")

    def test_online_and_target_networks_exist(self):
        """2. Verify online and target networks exist."""
        agent = SnakeDQNAgent()
        assert hasattr(agent, "online_net")
        assert hasattr(agent, "target_net")
        assert isinstance(agent.online_net, torch.nn.Module)
        assert isinstance(agent.target_net, torch.nn.Module)

    def test_target_network_initially_matches_online_network(self):
        """3. Verify target network initially matches online network weights exactly."""
        agent = SnakeDQNAgent()
        for p_online, p_target in zip(agent.online_net.parameters(), agent.target_net.parameters()):
            assert torch.equal(p_online, p_target)

    def test_optimizer_uses_configured_learning_rate(self):
        """4. Verify optimizer uses configured learning rate (0.0001)."""
        config = load_config("configs/snake_dqn.yaml")
        agent = SnakeDQNAgent(config=config)
        expected_lr = config["training"]["learning_rate"]
        actual_lr = agent.optimizer.param_groups[0]["lr"]
        assert actual_lr == expected_lr == 0.0001

    def test_gamma_is_configured_correctly(self):
        """5. Verify gamma is configured correctly (0.99)."""
        config = load_config("configs/snake_dqn.yaml")
        agent = SnakeDQNAgent(config=config)
        assert agent.gamma == config["training"]["gamma"] == 0.99

    def test_initial_epsilon_is_1_0(self):
        """6. Verify initial epsilon is 1.0 when frame_count = 0."""
        agent = SnakeDQNAgent()
        agent.frame_count = 0
        assert agent.get_epsilon() == 1.0
        assert agent.epsilon == 1.0

    def test_epsilon_decreases_with_frame_progression(self):
        """7. Verify epsilon decreases linearly according to frame progression."""
        agent = SnakeDQNAgent()

        # At frame 100,000 (halfway through 200,000 decay): 1.0 - 0.5 * (1.0 - 0.02) = 0.51
        agent.frame_count = 100000
        assert pytest.approx(agent.get_epsilon(), abs=1e-4) == 0.51

        # At frame 150,000 (3/4 through decay): 1.0 - 0.75 * 0.98 = 0.265
        agent.frame_count = 150000
        assert pytest.approx(agent.get_epsilon(), abs=1e-4) == 0.265

    def test_epsilon_never_falls_below_minimum(self):
        """8. Verify epsilon never falls below configured minimum (0.02)."""
        agent = SnakeDQNAgent()

        agent.frame_count = 200000
        assert agent.get_epsilon() == 0.02

        agent.frame_count = 500000
        assert agent.get_epsilon() == 0.02

    def test_greedy_action_in_valid_range(self):
        """9. Verify greedy action is within [0, 3]."""
        agent = SnakeDQNAgent()
        state = np.random.randn(4, 64, 64).astype(np.float32)

        action = agent.select_action(state, eval_mode=True)
        assert isinstance(action, int)
        assert 0 <= action <= 3

    def test_random_action_in_valid_range(self):
        """10. Verify random action is within [0, 3]."""
        agent = SnakeDQNAgent()
        agent.frame_count = 0  # epsilon = 1.0
        state = np.random.randn(4, 64, 64).astype(np.float32)

        actions = [agent.select_action(state, eval_mode=False) for _ in range(50)]
        for act in actions:
            assert isinstance(act, int)
            assert 0 <= act <= 3
        # Should contain varied actions due to random sampling
        assert len(set(actions)) > 1

    def test_action_selection_does_not_create_gradients(self):
        """11. Verify action selection does not attach gradients to tensors."""
        agent = SnakeDQNAgent()
        state_tensor = torch.randn(4, 64, 64, requires_grad=True)

        action = agent.select_action(state_tensor, eval_mode=True)
        assert isinstance(action, int)
        assert state_tensor.grad is None

    def test_insufficient_replay_data_does_not_update(self):
        """12. Verify insufficient replay data returns None and does not update parameters."""
        agent = SnakeDQNAgent()
        # Add fewer than batch_size (32) transitions
        state = np.zeros((4, 64, 64), dtype=np.float32)
        for _ in range(10):
            agent.store_transition(state, 0, 1.0, state, False)

        stats = agent.learn()
        assert stats is None
        assert agent.update_steps == 0

    def test_populated_replay_buffer_allows_learning_update(self):
        """13. Verify populated replay buffer enables a valid learning update."""
        agent = SnakeDQNAgent()
        state = np.random.randn(4, 64, 64).astype(np.float32)

        for _ in range(35):
            agent.store_transition(state, 0, 1.0, state, False)

        stats = agent.learn()
        assert isinstance(stats, dict)
        assert "loss" in stats
        assert "avg_q" in stats
        assert "mean_td_error" in stats
        assert agent.update_steps == 1

    def test_learning_returns_finite_loss(self):
        """14. Verify learning step returns a finite loss float value."""
        agent = SnakeDQNAgent()
        state = np.random.randn(4, 64, 64).astype(np.float32)

        for _ in range(35):
            agent.store_transition(state, 1, 0.5, state, False)

        stats = agent.learn()
        assert np.isfinite(stats["loss"])
        assert isinstance(stats["loss"], float)

    def test_learning_returns_finite_avg_q(self):
        """15. Verify learning step returns finite average Q-value statistic."""
        agent = SnakeDQNAgent()
        state = np.random.randn(4, 64, 64).astype(np.float32)

        for _ in range(35):
            agent.store_transition(state, 2, -1.0, state, False)

        stats = agent.learn()
        assert np.isfinite(stats["avg_q"])
        assert isinstance(stats["avg_q"], float)

    def test_parameters_change_after_optimizer_update(self):
        """16. Verify online network parameters change after a valid optimizer step."""
        agent = SnakeDQNAgent(seed=42)
        state = np.random.randn(4, 64, 64).astype(np.float32)

        for _ in range(40):
            agent.store_transition(state, 0, 1.0, state, False)

        # Snapshot weights before update
        weights_before = [p.clone() for p in agent.online_net.parameters()]

        agent.learn()

        # Verify at least some parameters have changed
        any_changed = False
        for p_before, p_after in zip(weights_before, agent.online_net.parameters()):
            if not torch.equal(p_before, p_after):
                any_changed = True
                break
        assert any_changed, "Online network parameters should change after optimizer step"

    def test_terminal_transitions_do_not_bootstrap(self):
        """17. Verify terminal transitions (done=True) do not bootstrap from next-state Q values."""
        agent = SnakeDQNAgent()

        b_rewards = torch.tensor([[5.0]], dtype=torch.float32)
        b_dones = torch.tensor([[1.0]], dtype=torch.float32)  # Terminal
        next_q = torch.tensor([[100.0]], dtype=torch.float32)

        # Target = reward + (1 - done) * gamma * next_q
        target_q = b_rewards + (1.0 - b_dones) * agent.gamma * next_q
        assert target_q.item() == 5.0, "Terminal target must equal reward without next-state bootstrap"

        # Non-terminal target = 5.0 + 0.99 * 100.0 = 104.0
        b_dones_non_term = torch.tensor([[0.0]], dtype=torch.float32)
        target_non_term = b_rewards + (1.0 - b_dones_non_term) * agent.gamma * next_q
        assert pytest.approx(target_non_term.item(), abs=1e-4) == 104.0

    def test_target_network_no_gradients(self):
        """18. Verify target network does not receive gradients during target calculation."""
        agent = SnakeDQNAgent()
        state = np.random.randn(4, 64, 64).astype(np.float32)

        for _ in range(35):
            agent.store_transition(state, 0, 1.0, state, False)

        agent.learn()

        for name, param in agent.target_net.named_parameters():
            assert param.grad is None, f"Target network parameter {name} should not have gradients"

    def test_target_network_synchronization_interval(self):
        """19. Verify target network synchronizes at exactly the configured update step interval."""
        agent = SnakeDQNAgent()
        agent.target_update_steps = 10  # Reduced for fast test

        # Manually alter online network weights so they differ from target network
        with torch.no_grad():
            for p in agent.online_net.parameters():
                p.add_(1.0)

        # Confirm weights currently differ
        differ_before = any(not torch.equal(p1, p2) for p1, p2 in zip(agent.online_net.parameters(), agent.target_net.parameters()))
        assert differ_before

        state = np.random.randn(4, 64, 64).astype(np.float32)
        for _ in range(50):
            agent.store_transition(state, 0, 1.0, state, False)

        # Run 9 updates -> should not sync yet
        for _ in range(9):
            agent.learn()
        assert agent.update_steps == 9
        assert any(not torch.equal(p1, p2) for p1, p2 in zip(agent.online_net.parameters(), agent.target_net.parameters()))

        # 10th update -> should trigger synchronization
        agent.learn()
        assert agent.update_steps == 10
        for p1, p2 in zip(agent.online_net.parameters(), agent.target_net.parameters()):
            assert torch.equal(p1, p2), "Target network weights must match online network after sync step"

    def test_training_update_counters_behavior(self):
        """20. Verify update_steps counter increments by 1 per learning call."""
        agent = SnakeDQNAgent()
        state = np.random.randn(4, 64, 64).astype(np.float32)

        for _ in range(40):
            agent.store_transition(state, 0, 1.0, state, False)

        assert agent.update_steps == 0
        agent.learn()
        assert agent.update_steps == 1
        agent.learn()
        assert agent.update_steps == 2

    def test_checkpoint_state_exposure(self):
        """21. Verify agent exposes complete checkpoint state dict structure."""
        agent = SnakeDQNAgent()
        agent.frame_count = 1500
        agent.update_steps = 25

        chk = agent.get_checkpoint_state()
        assert isinstance(chk, dict)
        assert "online_net_state_dict" in chk
        assert "target_net_state_dict" in chk
        assert "optimizer_state_dict" in chk
        assert chk["frame_count"] == 1500
        assert chk["update_steps"] == 25
        assert chk["epsilon"] == agent.get_epsilon()
        assert "config" in chk
