"""Tests for make_env environment factory function."""

import pytest
from src.environments import make_env
from src.environments.pacman.environment import PacManEnv
from src.environments.snake.environment import SnakeEnv
from src.environments.flappy.environment import FlappyEnv


class TestMakeEnvFactory:
    """Test suite for make_env factory function."""

    def test_make_pacman_env(self):
        """Verify make_env('pacman') returns PacManEnv instance with correct attributes."""
        env = make_env("pacman")
        assert isinstance(env, PacManEnv)
        assert env.num_actions == 9
        obs, _ = env.reset(seed=42)
        assert obs.shape == (84, 84, 1)
        env.close()

    def test_make_snake_env(self):
        """Verify make_env('snake') returns SnakeEnv instance with correct attributes."""
        env = make_env("snake")
        assert isinstance(env, SnakeEnv)
        assert env.num_actions == 4
        obs, _ = env.reset(seed=42)
        assert obs.shape == (84, 84, 1)
        env.close()

    def test_make_flappy_env(self):
        """Verify make_env('flappy') returns FlappyEnv instance with correct attributes."""
        env = make_env("flappy")
        assert isinstance(env, FlappyEnv)
        assert env.num_actions == 2
        obs, _ = env.reset(seed=42)
        assert obs.shape == (84, 84, 1)
        env.close()

    def test_unknown_game_raises_value_error(self):
        """Verify unknown game name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown game"):
            make_env("mario")

        with pytest.raises(ValueError, match="Unknown game"):
            make_env("tetris")
