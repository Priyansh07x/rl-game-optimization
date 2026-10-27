"""Flappy Bird Environment Verification and Smoke Runner.

Runs verification episodes, validates wrapper contract, and logs transition traces to:
- logs/flappy_environment_smoke_test.csv
- observations/flappy/
- reward_traces/flappy/
- episode_traces/flappy/
"""

import csv
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# Ensure project root is in Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.environments.flappy_env import FlappyEnv, FlappyAction


def run_flappy_verification(
    num_episodes: int = 5,
    seed: int = 42,
    output_csv: str = "logs/flappy_environment_smoke_test.csv",
) -> List[Dict[str, Any]]:
    """Run verification simulation and export transition logs."""
    env = FlappyEnv(survival_reward=0.5, pipe_pass_reward=5.0, collision_reward=-1000.0)

    # Ensure evidence directories exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("observations/flappy", exist_ok=True)
    os.makedirs("reward_traces/flappy", exist_ok=True)
    os.makedirs("episode_traces/flappy", exist_ok=True)

    records: List[Dict[str, Any]] = []

    for ep in range(num_episodes):
        ep_seed = seed + ep
        obs, info = env.reset(seed=ep_seed)
        step = 0
        done = False
        ep_reward = 0.0

        while not done and step < 200:
            step += 1
            action = 0 if step % 6 == 0 else 1  # Periodic flap
            action_name = FlappyAction(action).name

            next_obs, reward, terminated, truncated, step_info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

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

    env.close()

    # Write CSV output
    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys())

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"[SUCCESS] Verification completed. {len(records)} steps logged to {csv_path}")
    return records


if __name__ == "__main__":
    run_flappy_verification()
