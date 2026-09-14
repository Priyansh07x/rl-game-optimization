"""Snake Refined DQN Training Entry Point."""

import argparse
import random
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import torch

from src.agents.dqn.agent import SnakeDQNAgent
from src.common.checkpoint import CheckpointManager
from src.common.config import load_config
from src.common.logger import get_logger
from src.common.metrics import MetricsLogger
from src.environments.snake_env import SnakeEnv
from src.preprocessing.frame_stack import FrameStack
from src.preprocessing.snake_preprocess import SnakePreprocessor


def parse_args():
    """Parse command-line arguments for Snake DQN training."""
    parser = argparse.ArgumentParser(description="Train Snake Refined DQN Agent")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/snake_dqn.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Total episodes to train (overrides config if provided)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Target device ('auto', 'cpu', 'cuda')",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/snake",
        help="Directory for logs and CSV metrics",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/snake",
        help="Directory for periodic checkpoints",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory for best model export",
    )
    return parser.parse_args()


def setup_seed(seed: int) -> None:
    """Set random seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_training(
    config_path: str = "configs/snake_dqn.yaml",
    episodes: Optional[int] = None,
    seed: int = 42,
    resume_path: Optional[str] = None,
    device: Optional[str] = None,
    log_dir: str = "logs/snake",
    checkpoint_dir: str = "checkpoints/snake",
    model_dir: str = "models",
    checkpoint_interval_override: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute Snake DQN training pipeline.

    Args:
        config_path: Path to YAML configuration file.
        episodes: Optional override for total training episodes.
        seed: Random seed.
        resume_path: Optional checkpoint file path to resume training.
        device: Device identifier ('auto', 'cpu', 'cuda').
        log_dir: Directory path for logs and metrics CSV.
        checkpoint_dir: Directory path for periodic checkpoints.
        model_dir: Directory path for best model export.
        checkpoint_interval_override: Optional episode interval for saving checkpoints.

    Returns:
        Summary dict containing training stats and output paths.
    """
    setup_seed(seed)

    # Load configuration
    config = load_config(config_path)

    # Resolve training parameters
    total_episodes = episodes if episodes is not None else int(config.get("evaluation", {}).get("episodes", 100))
    checkpoint_interval = (
        checkpoint_interval_override
        if checkpoint_interval_override is not None
        else int(config.get("checkpoint", {}).get("interval_episodes", 100))
    )
    training_gap = int(config.get("training", {}).get("training_gap", 4))

    # Setup output directories
    log_path = Path(log_dir)
    chk_path = Path(checkpoint_dir)
    mdl_path = Path(model_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    chk_path.mkdir(parents=True, exist_ok=True)
    mdl_path.mkdir(parents=True, exist_ok=True)

    logger = get_logger(name="snake_dqn_training", log_dir=log_path)
    logger.info(f"Starting Snake DQN Training | Seed: {seed} | Device: {device or 'auto'}")

    # Environment & preprocessing setup
    env = SnakeEnv(obs_type="rgb", random_spawn=True)
    preprocessor = SnakePreprocessor(target_shape=(64, 64), dtype=np.float32)
    frame_stack = FrameStack(stack_size=4, frame_shape=(64, 64), dtype=np.float32)

    # Initialize agent
    agent = SnakeDQNAgent(config=config, device=device, seed=seed)

    # Initialize Checkpoint & Metrics managers
    checkpoint_manager = CheckpointManager(checkpoint_dir=chk_path)
    metrics_csv_path = log_path / "training_metrics.csv"
    metrics_logger = MetricsLogger(log_path=metrics_csv_path, append=True)

    start_episode = 1
    best_reward = -float("inf")
    best_score = -1.0
    best_model_path = mdl_path / "snake_dqn_best.pt"

    # Resume from checkpoint if requested
    if resume_path is not None:
        logger.info(f"Resuming training from checkpoint: {resume_path}")
        loaded_data = checkpoint_manager.load(resume_path, agent=agent)
        start_episode = int(loaded_data.get("episode", 0)) + 1
        logger.info(f"Resumed at Episode {start_episode} | Frame: {agent.frame_count} | Steps: {agent.update_steps}")

    for ep in range(start_episode, start_episode + total_episodes):
        ep_seed = seed + ep
        raw_obs, info = env.reset(seed=ep_seed)
        proc_frame = preprocessor.transform(raw_obs)
        stacked_state = frame_stack.reset(proc_frame)

        ep_reward = 0.0
        ep_length = 0
        ep_losses = []
        ep_qs = []
        done = False

        while not done:
            agent.frame_count += 1
            action = agent.select_action(stacked_state)

            next_raw, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            ep_length += 1
            done = terminated or truncated

            next_proc = preprocessor.transform(next_raw)
            next_stacked = frame_stack.step(next_proc)

            # Store transition
            agent.store_transition(
                state=stacked_state,
                action=action,
                reward=reward,
                next_state=next_stacked,
                done=done,
            )

            stacked_state = next_stacked

            # Perform learning update every K (training_gap) frames
            if agent.frame_count % training_gap == 0:
                learn_stats = agent.learn()
                if learn_stats is not None:
                    ep_losses.append(learn_stats["loss"])
                    ep_qs.append(learn_stats["avg_q"])

        # Compute summary episode metrics
        mean_loss = float(np.mean(ep_losses)) if ep_losses else None
        mean_q = float(np.mean(ep_qs)) if ep_qs else None
        score = float(info.get("score", 0))

        # Log episode metrics
        metrics_logger.log_episode(
            episode=ep,
            reward=ep_reward,
            episode_length=ep_length,
            loss=mean_loss,
            epsilon=agent.get_epsilon(),
            avg_q=mean_q,
            score=score,
            seed=seed,
        )

        logger.info(
            f"Episode {ep}/{start_episode + total_episodes - 1} | "
            f"Reward: {ep_reward:.1f} | Length: {ep_length} | Score: {int(score)} | "
            f"Eps: {agent.get_epsilon():.3f} | Loss: {f'{mean_loss:.4f}' if mean_loss is not None else 'N/A'}"
        )

        # Check best model
        if ep_reward > best_reward:
            best_reward = ep_reward
            best_score = score
            checkpoint_manager.save(
                agent=agent,
                episode=ep,
                is_best=True,
                best_filename=best_model_path,
                extra_info={"reward": best_reward, "score": best_score},
            )
            logger.info(f"New best model saved to {best_model_path} (Reward: {best_reward:.1f})")

        # Save periodic checkpoint
        if ep % checkpoint_interval == 0:
            saved_file = checkpoint_manager.save(agent=agent, episode=ep)
            logger.info(f"Periodic checkpoint saved: {saved_file}")

    metrics_logger.close()
    env.close()

    return {
        "start_episode": start_episode,
        "end_episode": start_episode + total_episodes - 1,
        "total_episodes_run": total_episodes,
        "final_frame_count": agent.frame_count,
        "final_update_steps": agent.update_steps,
        "best_reward": best_reward,
        "best_score": best_score,
        "best_model_path": str(best_model_path),
        "metrics_csv_path": str(metrics_csv_path),
    }


def main():
    args = parse_args()
    run_training(
        config_path=args.config,
        episodes=args.episodes,
        seed=args.seed,
        resume_path=args.resume,
        device=args.device,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        model_dir=args.model_dir,
    )


if __name__ == "__main__":
    main()
