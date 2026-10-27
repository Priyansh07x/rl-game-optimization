"""General DQN training entrypoint for Pac-Man, Snake, and Flappy Bird."""

import argparse
import csv
from pathlib import Path
import time
import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.environments import make_env
from src.agents.dqn.pacman_dqn import PacManDQNAgent


def train(game: str, config_path: str = None, episodes: int = 50, seed: int = 42, resume: str = None):
    """Train DQN agent on the specified game.

    Args:
        game: Name of game ('pacman', 'snake', 'flappy').
        config_path: Path to YAML config file. If None, defaults to configs/{game}_dqn.yaml.
        episodes: Number of episodes to train.
        seed: Random seed.
        resume: Path to checkpoint to resume training from.
    """
    if config_path is None:
        config_path = f"configs/{game}_dqn.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    np.random.seed(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    env = make_env(game)
    agent = PacManDQNAgent(
        num_actions=env.num_actions,
        lr=float(config["training"]["learning_rate"]),
        gamma=float(config["training"]["gamma"]),
        device=device,
    )

    start_episode = 0
    if resume:
        start_episode = agent.load(resume)
        print(f"Resumed from episode {start_episode}")

    log_path = Path(config["logging"]["log_file"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(config["checkpoint"]["save_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = models_dir / f"{game}_dqn_best.pt"
    legacy_best_path = ckpt_dir.parent / f"{game}_dqn_best.pt"

    csv_file = open(log_path, "a" if resume else "w", newline="", encoding="utf-8")
    fields = ["episode", "reward", "episode_length", "loss", "epsilon"]
    writer = csv.DictWriter(csv_file, fieldnames=fields)
    if not resume:
        writer.writeheader()

    best_avg = -np.inf
    recent = []
    t0 = time.time()
    ckpt_freq = config["checkpoint"].get("frequency", config["checkpoint"].get("interval_episodes", 100))

    for ep in tqdm(range(start_episode, start_episode + episodes),
                   desc=f"Training {game.capitalize()}", unit="episode", total=episodes):
        obs, _ = env.reset(seed=seed + ep)
        state = np.zeros((4, 84, 84), dtype=np.float32)
        state[-1] = obs.squeeze()

        ep_reward, ep_steps, ep_losses = 0.0, 0, []
        done = False
        while not done:
            action = agent.select_action(state)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_state = np.roll(state, -1, axis=0)
            next_state[-1] = obs.squeeze()

            loss = agent.update(state, action, reward, next_state, done)
            if loss is not None:
                ep_losses.append(loss)

            ep_reward += reward
            ep_steps += 1
            state = next_state

        writer.writerow({
            "episode": ep + 1,
            "reward": ep_reward,
            "episode_length": ep_steps,
            "loss": float(np.mean(ep_losses)) if ep_losses else 0.0,
            "epsilon": agent.epsilon,
        })
        csv_file.flush()

        recent.append(ep_reward)
        recent = recent[-50:]
        avg50 = float(np.mean(recent))

        # Save best model if 50-episode average improves or initial save
        if (len(recent) == 50 and avg50 > best_avg) or (not best_model_path.exists() and ep == start_episode):
            if len(recent) == 50:
                best_avg = avg50
            agent.save(str(best_model_path), episode=ep + 1)
            agent.save(str(legacy_best_path), episode=ep + 1)

        if (ep + 1) % 10 == 0:
            tqdm.write(f"ep {ep+1} | reward {ep_reward:.0f} | avg50 {avg50:.1f} | "
                       f"eps {agent.epsilon:.3f} | steps {agent.total_steps} | "
                       f"{(time.time()-t0)/60:.1f} min")

        if (ep + 1) % ckpt_freq == 0:
            agent.save(str(ckpt_dir / f"checkpoint_ep{ep+1}.pt"), episode=ep + 1)

    csv_file.close()
    final_model_path = models_dir / f"{game}_dqn_final.pt"
    agent.save(str(final_model_path), episode=start_episode + episodes)
    agent.save(str(ckpt_dir.parent / f"{game}_dqn_final.pt"), episode=start_episode + episodes)
    env.close()
    print(f"Done. Best 50-episode average: {best_avg:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN for Atari & Grid Games")
    parser.add_argument("--game", choices=["pacman", "snake", "flappy"], default="pacman")
    parser.add_argument("--config", default=None)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    train(
        game=args.game,
        config_path=args.config,
        episodes=args.episodes,
        seed=args.seed,
        resume=args.resume,
    )
