"""Runtime smoke and integration tests for the Snake environment.

Verifies end-to-end episode execution, observation space conformance, action execution,
reward validity, collision handling, truncation, post-termination resets, and logs
transition traces to logs/snake_environment_smoke_test.csv.
"""

import csv
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pytest

from src.environments.snake_env import SnakeAction, SnakeEnv


def run_smoke_episodes(
    env: SnakeEnv,
    num_episodes: int = 5,
    max_steps_per_episode: int = 100,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Execute smoke episodes and collect transition-level evidence.

    Args:
        env: SnakeEnv instance to test.
        num_episodes: Number of episodes to simulate.
        max_steps_per_episode: Maximum steps per episode before manual break.
        seed: Initial random seed.

    Returns:
        List of transition record dictionaries.
    """
    records: List[Dict[str, Any]] = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed + ep)
        assert env.observation_space.contains(obs), f"Initial obs not in observation_space: {obs}"
        assert isinstance(info, dict), "Reset info must be a dictionary"
        assert info["steps"] == 0
        assert info["score"] == 0
        assert info["snake_length"] == env.initial_length

        ep_reward = 0.0
        step = 0
        done = False

        while not done and step < max_steps_per_episode:
            step += 1
            # Sample legal action from action space
            action = int(env.action_space.sample())
            action_name = SnakeAction(action).name

            next_obs, reward, terminated, truncated, step_info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

            # Verifications for every transition
            assert env.observation_space.contains(next_obs), f"Step obs not in observation_space: {next_obs}"
            assert isinstance(reward, (int, float, np.floating)), f"Reward not numeric: {reward}"
            assert np.isfinite(reward), f"Reward not finite: {reward}"
            assert isinstance(terminated, bool), f"Terminated must be bool: {terminated}"
            assert isinstance(truncated, bool), f"Truncated must be bool: {truncated}"
            assert isinstance(step_info, dict), "Step info must be a dictionary"

            head = step_info["snake_head"]
            apple = step_info["apple"]

            record = {
                "episode": ep + 1,
                "step": step,
                "action": action,
                "action_name": action_name,
                "reward": float(reward),
                "cumulative_reward": float(ep_reward),
                "terminated": terminated,
                "truncated": truncated,
                "score": step_info["score"],
                "snake_length": step_info["snake_length"],
                "head_x": head[0] if head else -1,
                "head_y": head[1] if head else -1,
                "apple_x": apple[0] if apple else -1,
                "apple_y": apple[1] if apple else -1,
                "collision": step_info["collision"] if step_info["collision"] else "none",
            }
            records.append(record)

        # Confirm episode ended or completed steps
        assert done or step == max_steps_per_episode

    return records


def write_csv_evidence(records: List[Dict[str, Any]], output_path: Path) -> None:
    """Write transition records to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "episode",
        "step",
        "action",
        "action_name",
        "reward",
        "cumulative_reward",
        "terminated",
        "truncated",
        "score",
        "snake_length",
        "head_x",
        "head_y",
        "apple_x",
        "apple_y",
        "collision",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


class TestSnakeEnvSmokeIntegration:
    """Smoke and integration tests for Snake environment runtime behavior."""

    def test_grid_smoke_simulation_and_persistence(self):
        """Run multiple episodes with grid observation and verify evidence logging."""
        env = SnakeEnv(
            grid_size=12,
            map_size=240,
            initial_length=3,
            random_spawn=True,
            max_steps=100,
            obs_type="grid",
        )

        records = run_smoke_episodes(env, num_episodes=10, max_steps_per_episode=100, seed=1000)
        assert len(records) > 0, "Smoke simulation must record transitions"

        # Check that we have valid transitions across episodes
        episodes_seen = {r["episode"] for r in records}
        assert episodes_seen == set(range(1, 11))

        # Write to primary persistent CSV location
        csv_path = Path("logs/snake_environment_smoke_test.csv")
        write_csv_evidence(records, csv_path)
        assert csv_path.is_file(), f"CSV evidence file was not created at {csv_path}"
        assert csv_path.stat().st_size > 0, "CSV evidence file must not be empty"

    def test_rgb_smoke_simulation(self):
        """Run episodes with RGB observation representation."""
        env = SnakeEnv(
            grid_size=12,
            map_size=240,
            initial_length=3,
            random_spawn=True,
            max_steps=50,
            obs_type="rgb",
        )

        records = run_smoke_episodes(env, num_episodes=3, max_steps_per_episode=50, seed=2000)
        assert len(records) > 0
        for r in records:
            assert np.isfinite(r["reward"])

    def test_reset_after_episode_termination(self):
        """Explicitly verify that env cleanly resets and restarts after collision termination."""
        env = SnakeEnv(grid_size=6, initial_length=3, random_spawn=False)
        obs, info = env.reset(seed=42)

        # Drive snake straight into wall
        terminated = False
        steps = 0
        while not terminated and steps < 20:
            steps += 1
            obs, reward, terminated, truncated, info = env.step(SnakeAction.RIGHT)

        assert terminated is True
        assert info["collision"] == "wall_collision"

        # Attempt to reset
        new_obs, new_info = env.reset(seed=99)
        assert env.observation_space.contains(new_obs)
        assert new_info["score"] == 0
        assert new_info["steps"] == 0
        assert new_info["collision"] is None
        assert not env.terminated
        assert not env.truncated

        # Verify new steps proceed normally
        step_obs, step_reward, step_term, step_trunc, step_info = env.step(SnakeAction.UP)
        assert not step_term
        assert step_info["steps"] == 1

    def test_deterministic_reproducibility_smoke(self):
        """Verify identical action trajectories produce identical transitions with same seed."""
        env1 = SnakeEnv(grid_size=12, random_spawn=True)
        env2 = SnakeEnv(grid_size=12, random_spawn=True)

        env1.reset(seed=777)
        env2.reset(seed=777)

        actions = [SnakeAction.UP, SnakeAction.RIGHT, SnakeAction.DOWN, SnakeAction.RIGHT, SnakeAction.UP]
        for act in actions:
            obs1, r1, term1, trunc1, info1 = env1.step(act)
            obs2, r2, term2, trunc2, info2 = env2.step(act)

            assert np.array_equal(obs1, obs2)
            assert r1 == r2
            assert term1 == term2
            assert trunc1 == trunc2
            assert info1["snake_head"] == info2["snake_head"]
            assert info1["snake_body"] == info2["snake_body"]
            assert info1["apple"] == info2["apple"]

            if term1 or trunc1:
                break
