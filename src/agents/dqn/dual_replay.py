"""Dual Replay Buffer implementation for Snake Refined DQN."""

from collections import deque
from typing import Any, Optional, Tuple
import numpy as np


class DualReplayBuffer:
    """Dual-partition experience replay buffer split by TD-error magnitude.

    Configured according to Official Milestone 3 Snake Refined DQN specifications:
        - High-TD-error partition: capacity = 35,000 (|TD| > 0.5)
        - Low-TD-error partition: capacity = 15,000 (|TD| <= 0.5)
        - Total capacity = 50,000
        - Target sampling fraction: 70% High-TD, 30% Low-TD
    """

    def __init__(
        self,
        high_capacity: int = 35000,
        low_capacity: int = 15000,
        td_threshold: float = 0.5,
        high_fraction: float = 0.70,
        seed: Optional[int] = None,
    ):
        """Initialize DualReplayBuffer.

        Args:
            high_capacity: Max capacity of high-TD-error partition (default: 35000).
            low_capacity: Max capacity of low-TD-error partition (default: 15000).
            td_threshold: Absolute TD-error threshold for partition routing (default: 0.5).
            high_fraction: Target proportion of high-TD-error samples in minibatch (default: 0.70).
            seed: Random seed for reproducible sampling.
        """
        self.high_capacity = high_capacity
        self.low_capacity = low_capacity
        self.td_threshold = td_threshold
        self.high_fraction = high_fraction

        self.high_buffer: deque = deque(maxlen=high_capacity)
        self.low_buffer: deque = deque(maxlen=low_capacity)

        self.rng = np.random.default_rng(seed)

    def set_seed(self, seed: Optional[int]) -> None:
        """Set random seed for reproducible sampling."""
        self.rng = np.random.default_rng(seed)

    def add(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool,
        td_error: float,
    ) -> None:
        """Add a transition tuple to the appropriate buffer based on absolute TD-error.

        Args:
            state: Environment observation/state.
            action: Action taken.
            reward: Reward received.
            next_state: Next environment observation/state.
            done: Termination/truncation boolean flag.
            td_error: Scalar TD-error float value.
        """
        state_copy = np.array(state, copy=True) if isinstance(state, (np.ndarray, list)) else state
        next_state_copy = (
            np.array(next_state, copy=True) if isinstance(next_state, (np.ndarray, list)) else next_state
        )

        transition = (state_copy, int(action), float(reward), next_state_copy, bool(done))

        if abs(td_error) > self.td_threshold:
            self.high_buffer.append(transition)
        else:
            self.low_buffer.append(transition)

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a minibatch of transitions using the 70/30 target ratio with graceful fallback.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) batched numpy arrays.

        Raises:
            ValueError: If total available transitions across both partitions is less than batch_size.
        """
        total_available = len(self)
        if total_available < batch_size:
            raise ValueError(
                f"Cannot sample batch of size {batch_size}; only {total_available} total transitions available."
            )

        target_high = int(round(batch_size * self.high_fraction))
        target_low = batch_size - target_high

        len_high = len(self.high_buffer)
        len_low = len(self.low_buffer)

        # Determine partition sample counts with fallback
        if len_high >= target_high and len_low >= target_low:
            n_high = target_high
            n_low = target_low
        elif len_high < target_high:
            n_high = len_high
            n_low = batch_size - n_high
        else:  # len_low < target_low
            n_low = len_low
            n_high = batch_size - n_low

        high_indices = self.rng.choice(len_high, size=n_high, replace=False) if n_high > 0 else []
        low_indices = self.rng.choice(len_low, size=n_low, replace=False) if n_low > 0 else []

        high_samples = [self.high_buffer[i] for i in high_indices]
        low_samples = [self.low_buffer[i] for i in low_indices]

        sampled_transitions = high_samples + low_samples
        self.rng.shuffle(sampled_transitions)

        states = np.array([t[0] for t in sampled_transitions])
        actions = np.array([t[1] for t in sampled_transitions], dtype=np.int64)
        rewards = np.array([t[2] for t in sampled_transitions], dtype=np.float32)
        next_states = np.array([t[3] for t in sampled_transitions])
        dones = np.array([t[4] for t in sampled_transitions], dtype=np.bool_)

        return states, actions, rewards, next_states, dones

    @property
    def high_size(self) -> int:
        """Current number of transitions in high-TD partition."""
        return len(self.high_buffer)

    @property
    def low_size(self) -> int:
        """Current number of transitions in low-TD partition."""
        return len(self.low_buffer)

    @property
    def total_size(self) -> int:
        """Total number of transitions across both partitions."""
        return len(self.high_buffer) + len(self.low_buffer)

    def __len__(self) -> int:
        return self.total_size
