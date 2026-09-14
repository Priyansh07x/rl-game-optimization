"""Unit tests for Snake Gymnasium environment."""

import pytest
import numpy as np
from gymnasium import spaces

from src.environments.snake_env import (
    SnakeAction,
    SnakeEnv,
    ACTION_DIRECTIONS,
    OPPOSITE_ACTIONS,
)


class TestSnakeActionMapping:
    """Test explicit centralized action definitions."""

    def test_action_enum_values(self):
        assert SnakeAction.UP == 0
        assert SnakeAction.DOWN == 1
        assert SnakeAction.LEFT == 2
        assert SnakeAction.RIGHT == 3

    def test_action_directions(self):
        assert ACTION_DIRECTIONS[SnakeAction.UP] == (0, -1)
        assert ACTION_DIRECTIONS[SnakeAction.DOWN] == (0, 1)
        assert ACTION_DIRECTIONS[SnakeAction.LEFT] == (-1, 0)
        assert ACTION_DIRECTIONS[SnakeAction.RIGHT] == (1, 0)

    def test_opposite_actions(self):
        assert OPPOSITE_ACTIONS[SnakeAction.UP] == SnakeAction.DOWN
        assert OPPOSITE_ACTIONS[SnakeAction.DOWN] == SnakeAction.UP
        assert OPPOSITE_ACTIONS[SnakeAction.LEFT] == SnakeAction.RIGHT
        assert OPPOSITE_ACTIONS[SnakeAction.RIGHT] == SnakeAction.LEFT


class TestSnakeEnvInitialization:
    """Test environment initialization and spaces."""

    def test_default_initialization(self):
        env = SnakeEnv()
        assert env.grid_size == 12
        assert env.map_size == 240
        assert env.cell_size == 20
        assert env.initial_length == 3
        assert env.initial_direction == SnakeAction.RIGHT
        assert env.apple_reward == 1.0
        assert env.collision_reward == -1.0
        assert env.step_reward == 0.0
        assert isinstance(env.action_space, spaces.Discrete)
        assert env.action_space.n == 4

    def test_grid_observation_space(self):
        env = SnakeEnv(obs_type="grid")
        assert isinstance(env.observation_space, spaces.Box)
        assert env.observation_space.shape == (12, 12)
        assert env.observation_space.dtype == np.int8

    def test_rgb_observation_space(self):
        env = SnakeEnv(obs_type="rgb")
        assert isinstance(env.observation_space, spaces.Box)
        assert env.observation_space.shape == (240, 240, 3)
        assert env.observation_space.dtype == np.uint8


class TestSnakeEnvReset:
    """Test reset contract, initial length, direction, and seed reproducibility."""

    def test_reset_contract(self):
        env = SnakeEnv()
        obs, info = env.reset(seed=42)

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (12, 12)
        assert isinstance(info, dict)
        assert info["score"] == 0
        assert info["steps"] == 0
        assert info["snake_length"] == 3
        assert info["direction"] == "RIGHT"
        assert info["apple"] is not None
        assert info["collision"] is None

    def test_initial_snake_placement(self):
        env = SnakeEnv(random_spawn=False)
        _, info = env.reset()

        snake = info["snake_body"]
        assert len(snake) == 3
        # Head at (4, 6), body segments at (3, 6), (2, 6)
        assert snake[0] == (4, 6)
        assert snake[1] == (3, 6)
        assert snake[2] == (2, 6)

        # Apple must not overlap snake
        apple = info["apple"]
        assert apple not in snake
        assert 0 <= apple[0] < 12
        assert 0 <= apple[1] < 12

    def test_seed_reproducibility(self):
        env1 = SnakeEnv(random_spawn=True)
        env2 = SnakeEnv(random_spawn=True)

        obs1, info1 = env1.reset(seed=123)
        obs2, info2 = env2.reset(seed=123)

        assert np.array_equal(obs1, obs2)
        assert info1["snake_body"] == info2["snake_body"]
        assert info1["apple"] == info2["apple"]


class TestSnakeEnvMovement:
    """Test movement in all 4 directions and reverse direction prevention."""

    def test_move_right(self):
        env = SnakeEnv(random_spawn=False)
        env.reset()
        head_before = env.snake[0]

        obs, reward, terminated, truncated, info = env.step(SnakeAction.RIGHT)
        assert not terminated
        assert not truncated
        assert reward == 0.0
        assert env.snake[0] == (head_before[0] + 1, head_before[1])
        assert len(env.snake) == 3

    def test_move_up_and_left(self):
        env = SnakeEnv(random_spawn=False)
        env.reset()

        # Step UP
        obs, reward, terminated, truncated, info = env.step(SnakeAction.UP)
        assert not terminated
        assert env.direction == SnakeAction.UP
        assert env.snake[0] == (4, 5)

        # Step LEFT
        obs, reward, terminated, truncated, info = env.step(SnakeAction.LEFT)
        assert not terminated
        assert env.direction == SnakeAction.LEFT
        assert env.snake[0] == (3, 5)

    def test_move_down(self):
        env = SnakeEnv(random_spawn=False)
        env.reset()

        # Step DOWN
        obs, reward, terminated, truncated, info = env.step(SnakeAction.DOWN)
        assert not terminated
        assert env.direction == SnakeAction.DOWN
        assert env.snake[0] == (4, 7)

    def test_ignore_opposite_direction(self):
        """When moving RIGHT, pressing LEFT should be ignored and snake continues RIGHT."""
        env = SnakeEnv(random_spawn=False, allow_reverse=False)
        env.reset()
        head_before = env.snake[0]

        obs, reward, terminated, truncated, info = env.step(SnakeAction.LEFT)
        assert not terminated
        # Direction remains RIGHT, advances right
        assert env.direction == SnakeAction.RIGHT
        assert env.snake[0] == (head_before[0] + 1, head_before[1])


class TestSnakeEnvAppleAndGrowth:
    """Test eating apple, score increment, body growth, and new apple spawning."""

    def test_eat_apple_and_grow(self):
        env = SnakeEnv(random_spawn=False, apple_reward=1.0)
        env.reset()

        # Place apple directly in front of head (head is at (4, 6), facing RIGHT)
        target_apple = (5, 6)
        env.apple = target_apple

        obs, reward, terminated, truncated, info = env.step(SnakeAction.RIGHT)

        assert not terminated
        assert reward == 1.0
        assert info["apple_eaten"] is True
        assert info["score"] == 1
        assert info["snake_length"] == 4
        assert len(env.snake) == 4
        assert env.snake[0] == target_apple
        assert env.apple != target_apple
        assert env.apple not in env.snake


class TestSnakeEnvCollisionsAndTermination:
    """Test wall collision, self collision, and episode termination."""

    def test_wall_collision_right(self):
        env = SnakeEnv(grid_size=6, initial_length=3, collision_reward=-1.0, random_spawn=False)
        env.reset()
        # Head at (4, 3) on a 6x6 grid. Moving right 2 times hits x=6 (wall)
        env.step(SnakeAction.RIGHT)  # head at (5, 3)
        obs, reward, terminated, truncated, info = env.step(SnakeAction.RIGHT)  # head tries x=6

        assert terminated is True
        assert reward == -1.0
        assert info["collision"] == "wall_collision"

    def test_wall_collision_top(self):
        env = SnakeEnv(grid_size=6, initial_length=3, collision_reward=-1.0, random_spawn=False)
        env.reset()
        # Head at (4, 3). Moving UP 4 times hits y=-1 (wall)
        env.step(SnakeAction.UP)  # (4, 2)
        env.step(SnakeAction.UP)  # (4, 1)
        env.step(SnakeAction.UP)  # (4, 0)
        obs, reward, terminated, truncated, info = env.step(SnakeAction.UP)  # y=-1

        assert terminated is True
        assert reward == -1.0
        assert info["collision"] == "wall_collision"

    def test_self_collision(self):
        """Build a long snake and loop it into itself."""
        env = SnakeEnv(grid_size=12, initial_length=5, collision_reward=-1.0, random_spawn=False)
        env.reset()
        # Force snake body into a shape where turning around collides
        # Head at (5, 5), body [(5,5), (5,6), (4,6), (4,5), (4,4)]
        env.snake = [(5, 5), (5, 6), (4, 6), (4, 5), (4, 4)]
        env.direction = SnakeAction.UP

        # Action LEFT -> tries to move to (4, 5) which is body segment
        obs, reward, terminated, truncated, info = env.step(SnakeAction.LEFT)

        assert terminated is True
        assert reward == -1.0
        assert info["collision"] == "self_collision"

    def test_step_after_termination_raises_error(self):
        env = SnakeEnv(grid_size=6, initial_length=3, random_spawn=False)
        env.reset()
        env.step(SnakeAction.RIGHT)
        env.step(SnakeAction.RIGHT)  # terminates

        with pytest.raises(RuntimeError, match="Cannot call step\\(\\)"):
            env.step(SnakeAction.RIGHT)


class TestSnakeEnvTruncation:
    """Test max steps truncation."""

    def test_truncation(self):
        env = SnakeEnv(max_steps=5, random_spawn=False)
        env.reset()

        # Alternate UP/DOWN with RIGHT steps safely
        # Start (4,6) -> UP (4,5) -> RIGHT (5,5) -> DOWN (5,6) -> RIGHT (6,6)
        env.step(SnakeAction.UP)
        env.step(SnakeAction.RIGHT)
        env.step(SnakeAction.DOWN)
        env.step(SnakeAction.RIGHT)
        obs, reward, terminated, truncated, info = env.step(SnakeAction.UP)

        assert truncated is True
        assert terminated is False
        assert info["steps"] == 5


class TestSnakeEnvRendering:
    """Test ansi and rgb rendering modes."""

    def test_ansi_rendering(self):
        env = SnakeEnv(grid_size=6, initial_length=3, render_mode="ansi", random_spawn=False)
        env.reset()
        output = env.render()
        assert isinstance(output, str)
        assert "H" in output  # Head
        assert "B" in output  # Body
        assert "A" in output  # Apple
        assert "+" in output  # Border

    def test_rgb_rendering(self):
        env = SnakeEnv(map_size=240, grid_size=12, render_mode="rgb_array")
        env.reset()
        img = env.render()
        assert isinstance(img, np.ndarray)
        assert img.shape == (240, 240, 3)
        assert img.dtype == np.uint8
