"""Tests for FlappyEnv gymnasium environment."""

import numpy as np
import pytest
from src.environments.flappy.environment import FlappyEnv


class TestFlappyEnv:
    """Test suite for FlappyEnv verifying observation shapes, action space, step return, and reproducibility."""

    def test_reset_shape_and_dtype(self):
        """Verify reset returns (84, 84, 1) float32 observation in range [0, 1]."""
        env = FlappyEnv()
        obs, info = env.reset(seed=42)

        assert obs.shape == (84, 84, 1), f"Expected (84, 84, 1), got {obs.shape}"
        assert obs.dtype == np.float32, f"Expected float32, got {obs.dtype}"
        assert obs.min() >= 0.0 and obs.max() <= 1.0, f"Observation values outside [0, 1]: [{obs.min()}, {obs.max()}]"
        assert "score" in info
        env.close()

    def test_action_space(self):
        """Verify action space has exactly 2 actions (0=NOOP, 1=FLAP)."""
        env = FlappyEnv()
        assert env.action_space.n == 2, f"Expected 2 actions, got {env.action_space.n}"
        assert env.num_actions == 2
        env.close()

    def test_step_returns_5_tuple(self):
        """Verify step returns a valid 5-tuple (obs, reward, terminated, truncated, info)."""
        env = FlappyEnv()
        env.reset(seed=42)

        res = env.step(0)
        assert isinstance(res, tuple) and len(res) == 5, f"Expected 5-tuple, got {type(res)} of length {len(res)}"
        obs, reward, terminated, truncated, info = res

        assert obs.shape == (84, 84, 1)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert "score" in info
        assert "episode_return" in info
        env.close()

    def test_seed_reproducibility(self):
        """Verify seed reproducibility of the initial observation."""
        env1 = FlappyEnv()
        env2 = FlappyEnv()

        obs1, info1 = env1.reset(seed=42)
        obs2, info2 = env2.reset(seed=42)

        assert np.allclose(obs1, obs2), "Observations with same seed should match"
        assert info1["score"] == info2["score"]
        env1.close()
        env2.close()
