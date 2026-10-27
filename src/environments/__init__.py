"""Environments module."""

from src.environments.base_env import BaseEnv
from src.environments.pacman.environment import PacManEnv
from src.environments.snake.environment import SnakeEnv
from src.environments.flappy.environment import FlappyEnv
from src.environments.snake_env import (
    SnakeAction,
    ACTION_DIRECTIONS,
    OPPOSITE_ACTIONS,
)


def make_env(game: str, render_mode=None):
    """Factory function returning the environment instance for the requested game.

    Args:
        game: Game identifier ('pacman', 'snake', or 'flappy').
        render_mode: Rendering mode (None or 'human').

    Returns:
        Environment instance.

    Raises:
        ValueError: If game identifier is unknown.
    """
    game_lower = game.lower()
    if game_lower == "pacman":
        env = PacManEnv(render_mode=render_mode)
    elif game_lower == "snake":
        env = SnakeEnv(render_mode=render_mode)
    elif game_lower == "flappy":
        env = FlappyEnv(render_mode=render_mode)
    else:
        raise ValueError(f"Unknown game: {game}. Expected one of 'pacman', 'snake', 'flappy'.")

    # Verify num_actions attribute is present
    assert hasattr(env, "num_actions"), f"Environment {game} missing num_actions attribute."

    # Shape check: Observation must be (84, 84, 1) to drop into DQN architecture
    obs, _ = env.reset(seed=0)
    assert obs.shape == (84, 84, 1), f"Expected observation shape (84, 84, 1), got {obs.shape}"

    return env


__all__ = [
    "make_env",
    "PacManEnv",
    "SnakeEnv",
    "FlappyEnv",
    "BaseEnv",
    "SnakeAction",
    "ACTION_DIRECTIONS",
    "OPPOSITE_ACTIONS",
]
