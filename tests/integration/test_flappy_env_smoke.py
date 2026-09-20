"""Integration smoke test for Flappy Bird Environment wrapper.

Verifies end-to-end episode execution, observation space conformance, action execution,
reward validity, state discretization (75 states), collision handling, and writes
transition evidence to logs/flappy_environment_smoke_test.csv.
"""

import csv
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pytest

from src.environments.flappy_env import FlappyEnv, FlappyAction


def run_flappy_smoke_episodes(
    env: FlappyEnv,
    num_episodes: int = 5,
    seed: int = 100,
) -> List[Dict[str, Any]]:
    """Execute smoke episodes and collect transition-level evidence.

    Args:
        env: FlappyEnv instance to test.
        num_episodes: Number of episodes to simulate.
        seed: Initial random seed.

    Returns:
        List of transition record dictionaries.
    """
    records: List[Dict[str, Any]] = []

    for ep in range(num_episodes):
        ep_seed = seed + ep
        obs, info = env.reset(seed=ep_seed)

        assert env.observation_space.contains(obs), f"Initial obs not in observation_space: {obs}"
        assert isinstance(info, dict), "Reset info must be a dictionary"
        assert info["steps"] == 0
        assert info["score"] == 0
        assert 0 <= info["state_index"] < 75

        ep_reward = 0.0
        step = 0
        done = False

        while not done and step < 200:
            step += 1
            # Sample action: FLAP (0) every 5 steps, otherwise NO_FLAP (1)
            action = 0 if step % 5 == 0 else 1
            action_name = FlappyAction(action).name

            next_obs, reward, terminated, truncated, step_info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

            # Verifications for every transition
            assert env.observation_space.contains(next_obs), f"Step obs not in observation_space: {next_obs}"
            assert isinstance(reward, (int, float, np.floating)), f"Reward not numeric: {reward}"
            assert np.isfinite(reward), f"Reward not finite: {reward}"
            assert reward in (0.5, 5.0, -1000.0), f"Unexpected reward value: {reward}"
            assert isinstance(terminated, bool), f"Terminated must be bool: {terminated}"
            assert isinstance(truncated, bool), f"Truncated must be bool: {truncated}"
            assert isinstance(step_info, dict), "Step info must be a dictionary"
            assert 0 <= step_info["state_index"] < 75, f"State index out of bounds: {step_info['state_index']}"

            record = {
                "episode": ep + 1,
                "step": step,
                "action": action,
                "action_name": action_name,
                "native_action": step_info["native_action"],
                "dx": round(float(step_info["dx"]), 4),
                "dy": round(float(step_info["dy"]), 4),
                "v_y": round(float(step_info["v_y"]), 4),
                "x_bin": step_info["x_bin"],
                "y_bin": step_info["y_bin"],
                "v_bin": step_info["v_bin"],
                "state_index": step_info["state_index"],
                "reward": float(reward),
                "cumulative_reward": round(float(ep_reward), 2),
                "score": step_info["score"],
                "terminated": terminated,
                "truncated": truncated,
                "seed": ep_seed,
            }
            records.append(record)

        assert done or step == 200

    return records


def write_flappy_csv_evidence(records: List[Dict[str, Any]], output_path: Path) -> None:
    """Write transition records to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "episode",
        "step",
        "action",
        "action_name",
        "native_action",
        "dx",
        "dy",
        "v_y",
        "x_bin",
        "y_bin",
        "v_bin",
        "state_index",
        "reward",
        "cumulative_reward",
        "score",
        "terminated",
        "truncated",
        "seed",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


class TestFlappyEnvSmokeIntegration:
    """Smoke and integration tests for Flappy Bird environment runtime behavior."""

    def test_flappy_smoke_simulation_and_evidence_logging(self):
        """Run multiple Flappy Bird episodes and verify evidence logging to CSV."""
        env = FlappyEnv(survival_reward=0.5, pipe_pass_reward=5.0, collision_reward=-1000.0)

        records = run_flappy_smoke_episodes(env, num_episodes=5, seed=100)
        assert len(records) > 0, "Smoke simulation must record transitions"

        episodes_seen = {r["episode"] for r in records}
        assert episodes_seen == set(range(1, 6))

        # Save to primary persistent CSV location
        csv_path = Path("logs/flappy_environment_smoke_test.csv")
        write_flappy_csv_evidence(records, csv_path)
        assert csv_path.is_file(), f"CSV evidence file was not created at {csv_path}"
        assert csv_path.stat().st_size > 0, "CSV evidence file must not be empty"

        env.close()

    def test_flappy_reset_after_termination(self):
        """Explicitly verify clean reset after collision termination."""
        env = FlappyEnv()
        obs, info = env.reset(seed=42)

        terminated = False
        steps = 0
        while not terminated and steps < 100:
            steps += 1
            obs, reward, terminated, truncated, info = env.step(1)  # NO_FLAP to hit ground

        assert terminated is True
        assert reward == -1000.0

        # Attempt reset
        new_obs, new_info = env.reset(seed=99)
        assert env.observation_space.contains(new_obs)
        assert new_info["score"] == 0
        assert new_info["steps"] == 0
        assert 0 <= new_info["state_index"] < 75
        assert not env.terminated

        env.close()
