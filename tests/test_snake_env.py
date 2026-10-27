"""Tests for SnakeEnv gymnasium environment."""

import numpy as np
import pytest
from src.environments.snake.environment import SnakeEnv


class TestSnakeEnv:
    """Test suite for SnakeEnv verifying observations, reversal, rewards, collisions, and seeds."""

    def test_reset_shape_dtype_range(self):
        """Verify reset returns (84, 84, 1) float32 observation in range [0, 1]."""
        env = SnakeEnv()
        obs, info = env.reset(seed=42)

        assert obs.shape == (84, 84, 1), f"Expected (84, 84, 1), got {obs.shape}"
        assert obs.dtype == np.float32, f"Expected float32, got {obs.dtype}"
        assert obs.min() >= 0.0 and obs.max() <= 1.0, f"Observation values outside [0, 1]: [{obs.min()}, {obs.max()}]"
        assert "score" in info
        assert "length" in info
        assert info["score"] == 0
        assert info["length"] == 3
        env.close()

    def test_action_180_reversal_ignored(self):
        """Verify 180-degree reversal action is ignored and direction is maintained."""
        env = SnakeEnv()
        env.reset(seed=42)

        # Initial direction is RIGHT (3). Opposite is LEFT (2).
        assert env.direction == 3
        head_before = env.snake[0]

        # Step LEFT - should be ignored, snake continues RIGHT
        obs, reward, terminated, truncated, info = env.step(2)
        assert env.direction == 3, f"Expected direction 3 (RIGHT), got {env.direction}"
        assert env.snake[0] == (head_before[0] + 1, head_before[1])
        assert not terminated

        # Now direction is RIGHT. Turn UP (0).
        env.step(0)
        assert env.direction == 0
        head_up = env.snake[0]

        # Try DOWN (1) - opposite of UP, should be ignored
        env.step(1)
        assert env.direction == 0
        assert env.snake[0] == (head_up[0], head_up[1] - 1)
        env.close()

    def test_apple_eaten_reward_and_length(self):
        """Verify apple eaten yields +1.0 reward and increases snake length by 1."""
        env = SnakeEnv()
        env.reset(seed=42)

        # Place apple immediately in front of snake head
        head_x, head_y = env.snake[0]
        env.apple = (head_x + 1, head_y)
        initial_length = len(env.snake)

        # Move RIGHT into apple
        obs, reward, terminated, truncated, info = env.step(3)

        assert reward == 1.0, f"Expected reward 1.0 on apple eaten, got {reward}"
        assert len(env.snake) == initial_length + 1, f"Expected length {initial_length + 1}, got {len(env.snake)}"
        assert info["score"] == 1
        assert info["length"] == initial_length + 1
        assert not terminated
        env.close()

    def test_wall_collision_terminated_and_reward(self):
        """Verify wall collision terminates episode with -1.0 reward."""
        env = SnakeEnv(grid_size=6)
        env.reset(seed=42)

        # Head starts at (3, 3) facing RIGHT on 6x6 grid. Moving right reaches wall at x=6.
        env.step(3)  # head at (4, 3)
        env.step(3)  # head at (5, 3)
        obs, reward, terminated, truncated, info = env.step(3)  # tries to reach (6, 3) -> wall!

        assert terminated is True, "Expected terminated=True on wall collision"
        assert reward == -1.0, f"Expected reward -1.0 on collision, got {reward}"
        env.close()

    def test_seed_reproducibility(self):
        """Verify same seed produces identical initial observations."""
        env1 = SnakeEnv()
        env2 = SnakeEnv()

        obs1, info1 = env1.reset(seed=42)
        obs2, info2 = env2.reset(seed=42)

        assert np.array_equal(obs1, obs2), "Observations with same seed should be identical"
        assert env1.apple == env2.apple
        assert env1.snake == env2.snake
        assert info1 == info2
        env1.close()
        env2.close()
