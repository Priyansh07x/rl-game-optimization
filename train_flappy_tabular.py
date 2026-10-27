"""Flappy Bird Tabular RL Training Entry Point (Q-Learning & SARSA).

Supports training Q-Learning and SARSA under identical environments,
state discretization (75 states), actions (2 actions), reward structure,
epsilon decay schedule, and deterministic seeding.
"""

import argparse
import copy
from pathlib import Path
import random
from typing import Any, Dict, Optional, Union
import numpy as np

from src.agents.tabular.base_tabular import BaseTabularAgent
from src.agents.tabular.checkpoint import TabularCheckpointManager
from src.agents.tabular.q_learning import QLearningAgent
from src.agents.tabular.sarsa import SARSAAgent
from src.common.config import load_config
from src.common.logger import get_logger
from src.common.metrics import MetricsLogger
from src.environments.flappy_env import FlappyEnv

# Explicit column schema for tabular metrics logging (Part 5 requirement)
TABULAR_METRICS_COLUMNS = [
    "episode",
    "reward",
    "episode_length",
    "epsilon",
    "score",
    "q_table_size",
    "td_error",
    "seed",
]


def parse_args():
    """Parse command-line arguments for Flappy Bird tabular RL training."""
    parser = argparse.ArgumentParser(
        description="Train Flappy Bird Tabular RL Agent (Q-Learning / SARSA)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/flappy_tabular.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        choices=["q_learning", "sarsa"],
        help="Tabular RL algorithm to train ('q_learning' or 'sarsa')",
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
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory for logs and CSV metrics",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory for periodic checkpoints",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Directory for best model export",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=None,
        help="Episode interval for saving periodic checkpoints",
    )
    return parser.parse_args()


def setup_seed(seed: int) -> None:
    """Set random seeds across Python and NumPy for reproducible initialization."""
    random.seed(seed)
    np.random.seed(seed)


def create_agent(
    algorithm: str,
    config: Dict[str, Any],
    seed: int,
) -> BaseTabularAgent:
    """Instantiate QLearningAgent or SARSAAgent based on algorithm and configuration.

    Args:
        algorithm: Algorithm identifier ('q_learning' or 'sarsa').
        config: Loaded configuration dictionary.
        seed: Random seed for agent exploration RNG.

    Returns:
        Configured BaseTabularAgent instance.

    Raises:
        ValueError: If algorithm is not supported.
    """
    algo = algorithm.lower().strip()
    agent_cfg = config.get("agent", {})
    eps_cfg = config.get("epsilon", {})

    num_states = int(agent_cfg.get("num_states", 75))
    num_actions = int(agent_cfg.get("num_actions", 2))
    alpha = float(agent_cfg.get("alpha", 0.1))
    gamma = float(agent_cfg.get("gamma", 0.99))
    epsilon_start = float(eps_cfg.get("start", 1.0))
    epsilon_end = float(eps_cfg.get("end", 0.05))
    decay_episodes = int(eps_cfg.get("decay_episodes", 5000))

    if algo == "q_learning":
        return QLearningAgent(
            num_states=num_states,
            num_actions=num_actions,
            alpha=alpha,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            decay_episodes=decay_episodes,
            seed=seed,
        )
    elif algo == "sarsa":
        return SARSAAgent(
            num_states=num_states,
            num_actions=num_actions,
            alpha=alpha,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            decay_episodes=decay_episodes,
            seed=seed,
        )
    else:
        raise ValueError(
            f"Unsupported algorithm '{algo}'. Expected 'q_learning' or 'sarsa'."
        )


def run_tabular_training(
    config_path: str = "configs/flappy_tabular.yaml",
    algorithm: Optional[str] = None,
    episodes: Optional[int] = None,
    seed: Optional[int] = None,
    resume_path: Optional[str] = None,
    log_dir: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    model_dir: Optional[str] = None,
    checkpoint_interval: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute tabular reinforcement learning training pipeline (Q-Learning / SARSA).

    Args:
        config_path: Path to YAML configuration file.
        algorithm: Optional override for algorithm ('q_learning' or 'sarsa').
        episodes: Optional override for total training episodes.
        seed: Optional override for random seed.
        resume_path: Optional checkpoint file path to resume training from.
        log_dir: Optional override for log and metrics output directory.
        checkpoint_dir: Optional override for periodic checkpoint directory.
        model_dir: Optional override for best model export directory.
        checkpoint_interval: Optional override for checkpoint save frequency.

    Returns:
        Summary dictionary containing training stats and output paths.
    """
    # 1. Load base configuration
    config = load_config(config_path)

    # 2. Resolve algorithm
    algo = (
        algorithm
        if algorithm is not None
        else config.get("agent", {}).get("algorithm", "q_learning")
    ).lower().strip()

    if algo not in ("q_learning", "sarsa"):
        raise ValueError(f"Invalid algorithm '{algo}'. Expected 'q_learning' or 'sarsa'.")

    # 3. Resolve hyperparameters & settings
    base_seed = int(seed if seed is not None else config.get("seed", 42))
    setup_seed(base_seed)

    total_episodes = int(
        episodes if episodes is not None else config.get("training", {}).get("episodes", 5000)
    )
    if total_episodes <= 0:
        raise ValueError(f"Total episodes must be > 0, got {total_episodes}")

    chk_interval = int(
        checkpoint_interval
        if checkpoint_interval is not None
        else config.get("checkpoint", {}).get("interval_episodes", 500)
    )

    # 4. Resolve output directory paths
    dir_cfg = config.get("directories", {})
    resolved_log_dir = Path(log_dir if log_dir is not None else dir_cfg.get("log_dir", "logs/flappy"))
    resolved_chk_dir = Path(
        checkpoint_dir
        if checkpoint_dir is not None
        else dir_cfg.get("checkpoint_dir", f"checkpoints/flappy/{algo}")
    )
    resolved_model_dir = Path(
        model_dir if model_dir is not None else dir_cfg.get("model_dir", "models")
    )

    resolved_log_dir.mkdir(parents=True, exist_ok=True)
    resolved_chk_dir.mkdir(parents=True, exist_ok=True)
    resolved_model_dir.mkdir(parents=True, exist_ok=True)

    # 5. Initialize Logger
    logger = get_logger(name=f"flappy_{algo}_training", log_dir=resolved_log_dir)
    logger.info(
        f"Starting Flappy Tabular Training | Algorithm: {algo} | Episodes: {total_episodes} | Seed: {base_seed}"
    )

    # 6. Initialize Environment
    env = FlappyEnv(
        survival_reward=0.5,
        pipe_pass_reward=5.0,
        collision_reward=-1000.0,
    )

    # 7. Initialize Tabular Checkpoint Manager
    checkpoint_manager = TabularCheckpointManager(checkpoint_dir=resolved_chk_dir)

    # 8. Instantiate Agent & handle Resume
    start_episode = 1
    if resume_path is not None:
        logger.info(f"Resuming training from checkpoint: {resume_path}")
        # Peek checkpoint to validate algorithm match
        chk_info = checkpoint_manager.load(resume_path, agent=None)
        chk_algo = chk_info.get("algorithm", "").lower()
        if algorithm is not None and algo != chk_algo:
            raise ValueError(
                f"Algorithm mismatch: requested '{algo}', but checkpoint contains '{chk_algo}'."
            )
        algo = chk_algo
        agent = create_agent(algorithm=algo, config=config, seed=base_seed)
        loaded_data = checkpoint_manager.load(
            resume_path,
            agent=agent,
            restore_rng=True,
            expected_algorithm=algo,
        )
        start_episode = int(loaded_data.get("episode", 0)) + 1
        logger.info(
            f"Successfully resumed {algo} from Episode {start_episode - 1} | Next Episode: {start_episode} | Epsilon: {agent.epsilon:.4f}"
        )
    else:
        agent = create_agent(algorithm=algo, config=config, seed=base_seed)

    # 9. Initialize Metrics Logger
    metrics_csv_path = resolved_log_dir / f"{algo}_training_metrics.csv"
    metrics_logger = MetricsLogger(
        log_path=metrics_csv_path,
        columns=TABULAR_METRICS_COLUMNS,
        append=True,
    )

    best_reward = -float("inf")
    best_score = -1.0

    # 10. Training Loop
    end_episode = start_episode + total_episodes - 1
    for ep in range(start_episode, start_episode + total_episodes):
        ep_seed = base_seed + ep
        raw_obs, info = env.reset(seed=ep_seed)
        state = info["state_index"]

        ep_reward = 0.0
        ep_length = 0
        ep_td_errors = []
        done = False

        if algo == "q_learning":
            # --- Off-Policy Q-Learning Loop ---
            while not done:
                action = agent.select_action(state)
                next_raw, reward, terminated, truncated, info = env.step(action)
                ep_reward += float(reward)
                ep_length += 1
                done = terminated or truncated
                next_state = info["state_index"]

                td_error = agent.update(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                )
                ep_td_errors.append(abs(td_error))
                state = next_state

        elif algo == "sarsa":
            # --- On-Policy SARSA Loop ---
            # Initial action selected using behavior policy
            action = agent.select_action(state)
            while not done:
                next_raw, reward, terminated, truncated, info = env.step(action)
                ep_reward += float(reward)
                ep_length += 1
                done = terminated or truncated
                next_state = info["state_index"]

                if done:
                    next_action = None
                else:
                    next_action = agent.select_action(next_state)

                td_error = agent.update(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    next_action=next_action,
                    done=done,
                )
                ep_td_errors.append(abs(td_error))
                state = next_state
                if not done:
                    action = next_action

        # Update exploration rate following authoritative schedule
        agent.decay_epsilon(episode=ep)

        mean_td_error = float(np.mean(ep_td_errors)) if ep_td_errors else 0.0
        score = float(info.get("score", 0))

        # Log metrics row
        metrics_logger.log_episode(
            episode=ep,
            reward=ep_reward,
            episode_length=ep_length,
            loss=mean_td_error,
            epsilon=agent.epsilon,
            avg_q=None,
            score=score,
            seed=base_seed,
            q_table_size=agent.q_table_size,
            td_error=mean_td_error,
        )

        logger.info(
            f"[{algo.upper()}] Episode {ep}/{end_episode} | "
            f"Reward: {ep_reward:.1f} | Length: {ep_length} | Score: {int(score)} | "
            f"Eps: {agent.epsilon:.3f} | Mean TD: {mean_td_error:.4f} | NonZero Q: {agent.non_zero_entries}"
        )

        # Track best training episode (for metadata / intermediate inspection only)
        if ep_reward > best_reward:
            best_reward = ep_reward
            best_score = score
            checkpoint_manager.save(
                agent=agent,
                episode=ep,
                algorithm=algo,
                config=config,
                seed=base_seed,
                filename=f"flappy_{algo}_best.pkl",
                extra_info={"reward": best_reward, "score": best_score},
            )

        # Periodic checkpoint
        if ep % chk_interval == 0:
            saved_chk = checkpoint_manager.save(
                agent=agent,
                episode=ep,
                algorithm=algo,
                config=config,
                seed=base_seed,
                filename=f"checkpoint_ep{ep}.pkl",
                extra_info={"reward": ep_reward, "score": score},
            )
            logger.info(f"Periodic checkpoint saved: {saved_chk}")

    metrics_logger.close()
    env.close()

    logger.info(
        f"Training complete for {algo} | Total Episodes Run: {total_episodes} | Final Epsilon: {agent.epsilon:.4f}"
    )

    return {
        "algorithm": algo,
        "start_episode": start_episode,
        "end_episode": end_episode,
        "total_episodes_run": total_episodes,
        "final_epsilon": agent.epsilon,
        "best_reward": best_reward,
        "best_score": best_score,
        "metrics_csv_path": str(metrics_csv_path),
        "q_table": agent.q_table.copy(),
        "non_zero_entries": agent.non_zero_entries,
    }


def main():
    """CLI entry point for Flappy Bird tabular RL training."""
    args = parse_args()
    run_tabular_training(
        config_path=args.config,
        algorithm=args.algorithm,
        episodes=args.episodes,
        seed=args.seed,
        resume_path=args.resume,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        model_dir=args.model_dir,
        checkpoint_interval=args.checkpoint_interval,
    )


if __name__ == "__main__":
    main()
