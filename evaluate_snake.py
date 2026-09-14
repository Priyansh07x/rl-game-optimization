"""Formal Snake DQN Evaluation Entry Point."""

import argparse
import csv
import random
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import torch

from src.agents.dqn.agent import SnakeDQNAgent
from src.common.checkpoint import CheckpointManager
from src.common.logger import get_logger
from src.environments.snake_env import SnakeEnv
from src.preprocessing.frame_stack import FrameStack
from src.preprocessing.snake_preprocess import SnakePreprocessor


def parse_args():
    """Parse command-line arguments for Snake DQN evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Trained Snake Refined DQN Agent")
    parser.add_argument(
        "--model",
        type=str,
        default="models/snake_dqn_best.pt",
        help="Path to trained model checkpoint file",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for evaluation reproducibility",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Target device ('auto', 'cpu', 'cuda')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs/snake/evaluation",
        help="Directory to save evaluation_metrics.csv",
    )
    return parser.parse_args()


def setup_seed(seed: int) -> None:
    """Set random seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_evaluation(
    model_path: str = "models/snake_dqn_best.pt",
    episodes: int = 100,
    seed: int = 42,
    device: str = "cpu",
    output_dir: str = "logs/snake/evaluation",
) -> Dict[str, Any]:
    """Run formal evaluation of trained Snake DQN agent.

    Args:
        model_path: Path to PyTorch model/checkpoint file.
        episodes: Number of evaluation episodes.
        seed: Random seed.
        device: Computing device identifier ('cpu', 'cuda', 'auto').
        output_dir: Directory path for saving evaluation_metrics.csv.

    Returns:
        Summary dict containing detailed evaluation statistics.
    """
    setup_seed(seed)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    logger = get_logger(name="snake_dqn_evaluation", log_dir=out_path)
    logger.info(
        f"Starting Formal Snake DQN Evaluation | Model: {model_path} | Episodes: {episodes} | Seed: {seed} | Device: {device}"
    )

    # Load agent and checkpoint
    agent = SnakeDQNAgent(device=device, seed=seed)
    checkpoint_manager = CheckpointManager()

    loaded_data = checkpoint_manager.load(model_path, agent=agent, map_location=device, restore_rng=False)
    logger.info(f"Successfully loaded model from {model_path} (Format version: {loaded_data.get('format_version')})")

    # Set networks to eval mode
    agent.online_net.eval()
    agent.target_net.eval()

    # Capture weight snapshot before evaluation for integrity check
    param_snapshot = [p.clone().detach() for p in agent.online_net.parameters()]

    # Environment & preprocessing setup
    env = SnakeEnv(obs_type="rgb", random_spawn=True)
    preprocessor = SnakePreprocessor(target_shape=(64, 64), dtype=np.float32)
    frame_stack = FrameStack(stack_size=4, frame_shape=(64, 64), dtype=np.float32)

    metrics_csv_path = out_path / "evaluation_metrics.csv"
    records: List[Dict[str, Any]] = []

    with torch.no_grad():
        for ep in range(1, episodes + 1):
            ep_seed = seed + ep
            raw_obs, info = env.reset(seed=ep_seed)
            proc_frame = preprocessor.transform(raw_obs)
            stacked_state = frame_stack.reset(proc_frame)

            ep_reward = 0.0
            ep_length = 0
            done = False

            while not done:
                # Deterministic greedy action selection (eval_mode=True -> epsilon = 0.0)
                action = agent.select_action(stacked_state, eval_mode=True)

                next_raw, reward, terminated, truncated, info = env.step(action)
                ep_reward += float(reward)
                ep_length += 1
                done = terminated or truncated

                next_proc = preprocessor.transform(next_raw)
                stacked_state = frame_stack.step(next_proc)

            score = float(info.get("score", 0))
            record = {
                "episode": ep,
                "reward": ep_reward,
                "episode_length": ep_length,
                "score": score,
                "seed": seed,
            }
            records.append(record)

    env.close()

    # Verify model weights remained completely untouched during evaluation
    param_intact = all(
        torch.equal(p1, p2)
        for p1, p2 in zip(param_snapshot, agent.online_net.parameters())
    )
    if not param_intact:
        raise RuntimeError("Model parameter integrity check failed: weights changed during evaluation!")
    logger.info("Model parameter integrity check PASSED: weights remained unchanged.")

    # Save evaluation CSV
    with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "reward", "episode_length", "score", "seed"])
        writer.writeheader()
        writer.writerows(records)
    logger.info(f"Evaluation metrics written to {metrics_csv_path}")

    # Compute summary statistics
    rewards = [r["reward"] for r in records]
    scores = [r["score"] for r in records]
    lengths = [r["episode_length"] for r in records]

    positive_score_count = sum(1 for s in scores if s > 0)
    positive_score_pct = (positive_score_count / episodes) * 100.0

    stats = {
        "model_path": str(model_path),
        "episodes": episodes,
        "seed": seed,
        "device": str(device),
        "epsilon": 0.0,
        "mean_reward": float(np.mean(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "median_reward": float(np.median(rewards)),
        "mean_score": float(np.mean(scores)),
        "min_score": float(np.min(scores)),
        "max_score": float(np.max(scores)),
        "positive_score_count": positive_score_count,
        "positive_score_pct": positive_score_pct,
        "mean_length": float(np.mean(lengths)),
        "min_length": int(np.min(lengths)),
        "max_length": int(np.max(lengths)),
        "model_integrity_passed": param_intact,
        "metrics_csv_path": str(metrics_csv_path),
    }

    return stats


def main():
    args = parse_args()
    stats = run_evaluation(
        model_path=args.model,
        episodes=args.episodes,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
    )
    print("\n--- Evaluation Complete ---")
    print(f"Episodes: {stats['episodes']}")
    print(f"Mean Reward: {stats['mean_reward']:.4f} (Min: {stats['min_reward']}, Max: {stats['max_reward']}, Median: {stats['median_reward']})")
    print(f"Mean Score: {stats['mean_score']:.4f} (Max: {stats['max_score']}, >0 Score Episodes: {stats['positive_score_count']}/{stats['episodes']} [{stats['positive_score_pct']:.1f}%])")
    print(f"Mean Length: {stats['mean_length']:.2f} (Min: {stats['min_length']}, Max: {stats['max_length']})")


if __name__ == "__main__":
    main()
