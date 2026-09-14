"""Unit tests for DualReplayBuffer implementation."""

import numpy as np
import pytest

from src.agents.dqn.dual_replay import DualReplayBuffer
from src.common.config import load_config


class TestDualReplayBuffer:
    """Test suite for DualReplayBuffer routing, capacity, sampling, and fallback logic."""

    def test_empty_buffer_behavior(self):
        """1. Verify empty buffer size reporting and behavior."""
        buf = DualReplayBuffer()
        assert buf.high_size == 0
        assert buf.low_size == 0
        assert buf.total_size == 0
        assert len(buf) == 0

    def test_low_error_routing(self):
        """2. Adding a low-error transition routes it to the low-error partition."""
        buf = DualReplayBuffer(td_threshold=0.5)
        state = np.zeros((4, 64, 64), dtype=np.float32)
        buf.add(state=state, action=1, reward=1.0, next_state=state, done=False, td_error=0.2)

        assert buf.low_size == 1
        assert buf.high_size == 0
        assert buf.total_size == 1

    def test_high_error_routing(self):
        """3. Adding a high-error transition routes it to the high-error partition."""
        buf = DualReplayBuffer(td_threshold=0.5)
        state = np.zeros((4, 64, 64), dtype=np.float32)
        buf.add(state=state, action=2, reward=0.0, next_state=state, done=False, td_error=0.8)

        assert buf.high_size == 1
        assert buf.low_size == 0
        assert buf.total_size == 1

    def test_td_threshold_boundary_behavior(self):
        """4. Verify boundary conditions at exactly td_error = 0.5 and just above 0.5."""
        buf = DualReplayBuffer(td_threshold=0.5)
        state = np.zeros((4, 64, 64), dtype=np.float32)

        # Exact boundary 0.5 -> low error (|TD| <= 0.5)
        buf.add(state, 0, 0.0, state, False, td_error=0.5)
        assert buf.low_size == 1
        assert buf.high_size == 0

        # Exact boundary -0.5 -> low error
        buf.add(state, 0, 0.0, state, False, td_error=-0.5)
        assert buf.low_size == 2
        assert buf.high_size == 0

        # Just above 0.5 -> high error (|TD| > 0.5)
        buf.add(state, 0, 0.0, state, False, td_error=0.50001)
        assert buf.high_size == 1
        assert buf.low_size == 2

        # Just below -0.5 -> high error
        buf.add(state, 0, 0.0, state, False, td_error=-0.50001)
        assert buf.high_size == 2
        assert buf.low_size == 2

    def test_high_partition_capacity_limit(self):
        """5. Verify high partition never exceeds high_capacity (e.g. 35,000)."""
        capacity = 100
        buf = DualReplayBuffer(high_capacity=capacity, low_capacity=50)
        state = np.zeros((4, 64, 64), dtype=np.float32)

        for i in range(capacity + 20):
            buf.add(state, 0, 0.0, state, False, td_error=1.0)

        assert buf.high_size == capacity
        assert buf.low_size == 0
        assert buf.total_size == capacity

    def test_low_partition_capacity_limit(self):
        """6. Verify low partition never exceeds low_capacity (e.g. 15,000)."""
        capacity = 50
        buf = DualReplayBuffer(high_capacity=100, low_capacity=capacity)
        state = np.zeros((4, 64, 64), dtype=np.float32)

        for i in range(capacity + 15):
            buf.add(state, 0, 0.0, state, False, td_error=0.1)

        assert buf.low_size == capacity
        assert buf.high_size == 0
        assert buf.total_size == capacity

    def test_total_capacity_limit(self):
        """7. Verify total capacity equals high_capacity + low_capacity (35,000 + 15,000 = 50,000)."""
        # Load official values from config
        config = load_config("configs/snake_dqn.yaml")
        high_cap = config["replay"]["high_error_capacity"]
        low_cap = config["replay"]["low_error_capacity"]
        total_cap = config["replay"]["total_capacity"]

        assert high_cap == 35000
        assert low_cap == 15000
        assert total_cap == 50000

        buf = DualReplayBuffer(high_capacity=high_cap, low_capacity=low_cap)

        # Scale down for fast execution while maintaining ratio logic
        mini_buf = DualReplayBuffer(high_capacity=350, low_capacity=150)
        state = np.zeros((4, 64, 64), dtype=np.float32)

        for _ in range(400):
            mini_buf.add(state, 0, 0.0, state, False, td_error=1.0)
        for _ in range(200):
            mini_buf.add(state, 0, 0.0, state, False, td_error=0.1)

        assert mini_buf.high_size == 350
        assert mini_buf.low_size == 150
        assert mini_buf.total_size == 500

    def test_sampling_batch_size(self):
        """8. Verify sampling returns requested batch size when enough data exists."""
        buf = DualReplayBuffer(seed=42)
        state = np.zeros((4, 64, 64), dtype=np.float32)

        for i in range(50):
            buf.add(state, 0, 1.0, state, False, td_error=1.0 if i % 2 == 0 else 0.1)

        states, actions, rewards, next_states, dones = buf.sample(batch_size=32)
        assert states.shape == (32, 4, 64, 64)
        assert actions.shape == (32,)
        assert rewards.shape == (32,)
        assert next_states.shape == (32, 4, 64, 64)
        assert dones.shape == (32,)

    def test_sampling_70_30_target_ratio_distribution(self):
        """9. Verify sampling approximately follows the 70/30 high/low target ratio."""
        buf = DualReplayBuffer(seed=12345, high_fraction=0.70)
        state = np.zeros((4, 64, 64), dtype=np.float32)

        # Add 100 high-error transitions (reward = 100.0) and 100 low-error transitions (reward = 1.0)
        for _ in range(100):
            buf.add(state, 1, 100.0, state, False, td_error=1.5)
        for _ in range(100):
            buf.add(state, 1, 1.0, state, False, td_error=0.1)

        # Perform 50 minibatch samples of size 100 and count high-error occurrences
        total_sampled = 0
        total_high = 0

        for _ in range(50):
            _, _, rewards, _, _ = buf.sample(batch_size=100)
            high_count = np.sum(rewards == 100.0)
            total_high += high_count
            total_sampled += 100

        observed_ratio = total_high / total_sampled
        # 70% target -> should be within 0.70 +/- 0.05
        assert 0.65 <= observed_ratio <= 0.75, f"Observed ratio {observed_ratio} deviates from 0.70"

    def test_fallback_when_one_partition_underpopulated(self):
        """10. Verify fallback behavior when one partition contains fewer transitions than requested."""
        buf = DualReplayBuffer(seed=42, high_fraction=0.70)
        state = np.zeros((4, 64, 64), dtype=np.float32)

        # Only 5 high-error transitions (reward = 100.0), 50 low-error transitions (reward = 1.0)
        for _ in range(5):
            buf.add(state, 0, 100.0, state, False, td_error=1.0)
        for _ in range(50):
            buf.add(state, 0, 1.0, state, False, td_error=0.1)

        # Sample batch of 32 (target: 22 high, 10 low)
        # High partition has only 5 -> samples 5 high + 27 low = 32 total
        _, _, rewards, _, _ = buf.sample(batch_size=32)
        assert len(rewards) == 32
        assert np.sum(rewards == 100.0) == 5
        assert np.sum(rewards == 1.0) == 27

        # Inverse fallback: only 3 low-error transitions (reward = 1.0), 50 high-error (reward = 100.0)
        buf_inv = DualReplayBuffer(seed=42, high_fraction=0.70)
        for _ in range(50):
            buf_inv.add(state, 0, 100.0, state, False, td_error=1.0)
        for _ in range(3):
            buf_inv.add(state, 0, 1.0, state, False, td_error=0.1)

        # Sample batch of 32 (target: 22 high, 10 low)
        # Low partition has only 3 -> samples 3 low + 29 high = 32 total
        _, _, rewards_inv, _, _ = buf_inv.sample(batch_size=32)
        assert len(rewards_inv) == 32
        assert np.sum(rewards_inv == 1.0) == 3
        assert np.sum(rewards_inv == 100.0) == 29

    def test_total_available_fewer_than_batch_size(self):
        """11. Verify ValueError is raised when total available transitions < batch size."""
        buf = DualReplayBuffer()
        state = np.zeros((4, 64, 64), dtype=np.float32)

        for _ in range(10):
            buf.add(state, 0, 1.0, state, False, td_error=0.8)

        with pytest.raises(ValueError, match="Cannot sample batch of size 32"):
            buf.sample(batch_size=32)

    def test_sampled_transitions_preserve_fields_and_integrity(self):
        """12. Verify sampled transitions preserve state, action, reward, next_state, and done values."""
        buf = DualReplayBuffer()
        state_in = np.full((4, 64, 64), 7.0, dtype=np.float32)
        next_state_in = np.full((4, 64, 64), 8.0, dtype=np.float32)

        buf.add(state=state_in, action=3, reward=1.5, next_state=next_state_in, done=True, td_error=0.9)

        # Mutate caller state arrays to verify safety/independence
        state_in.fill(0.0)
        next_state_in.fill(0.0)

        # Add low item to allow batch of 2
        buf.add(state=state_in, action=2, reward=0.0, next_state=next_state_in, done=False, td_error=0.1)

        states, actions, rewards, next_states, dones = buf.sample(batch_size=2)

        # Find the transition with action=3
        idx = int(np.where(actions == 3)[0][0])
        assert np.all(states[idx] == 7.0)
        assert actions[idx] == 3
        assert rewards[idx] == 1.5
        assert np.all(next_states[idx] == 8.0)
        assert bool(dones[idx]) is True

    def test_buffer_size_reporting(self):
        """13. Verify high_size, low_size, total_size, and len() reporting."""
        buf = DualReplayBuffer(high_capacity=100, low_capacity=100)
        state = np.zeros((4, 64, 64), dtype=np.float32)

        assert buf.high_size == 0
        assert buf.low_size == 0
        assert buf.total_size == 0
        assert len(buf) == 0

        # Add 3 high, 2 low
        for _ in range(3):
            buf.add(state, 0, 0.0, state, False, td_error=0.9)
        for _ in range(2):
            buf.add(state, 0, 0.0, state, False, td_error=0.1)

        assert buf.high_size == 3
        assert buf.low_size == 2
        assert buf.total_size == 5
        assert len(buf) == 5
