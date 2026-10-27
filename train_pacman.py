import argparse
import yaml
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
import csv
from datetime import datetime, timedelta
import time
from tqdm import tqdm

from src.environments.pacman.environment import PacManEnv
from src.agents.dqn.pacman_dqn import PacManDQN


import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.environments.pacman.environment import PacManEnv
from src.agents.dqn.pacman_dqn import PacManDQNAgent


def train_pacman(config_path, episodes=50, seed=42, resume=None):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    np.random.seed(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    env = PacManEnv()
    agent = PacManDQNAgent(num_actions=env.action_space.n, lr=config["training"]["learning_rate"],
                           gamma=config["training"]["gamma"], device=device)

    start_episode = 0
    if resume:
        start_episode = agent.load(resume)
        print(f"Resumed from episode {start_episode}")

    log_path = Path(config["logging"]["log_file"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(config["checkpoint"]["save_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    csv_file = open(log_path, "a" if resume else "w", newline="")
    fields = ["episode", "reward", "episode_length", "loss", "epsilon"]
    writer = csv.DictWriter(csv_file, fieldnames=fields)
    if not resume:
        writer.writeheader()

    best_avg = -np.inf
    recent = []
    t0 = time.time()

    for ep in tqdm(range(start_episode, start_episode + episodes),
                   desc="Training", unit="episode", total=episodes):
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
        if len(recent) == 50 and avg50 > best_avg:
            best_avg = avg50
            agent.save(str(ckpt_dir.parent / "pacman_dqn_best.pt"), episode=ep + 1)

        if (ep + 1) % 10 == 0:
            tqdm.write(f"ep {ep+1} | reward {ep_reward:.0f} | avg50 {avg50:.1f} | "
                       f"eps {agent.epsilon:.3f} | steps {agent.total_steps} | "
                       f"{(time.time()-t0)/60:.1f} min")

        if (ep + 1) % config["checkpoint"]["frequency"] == 0:
            agent.save(str(ckpt_dir / f"checkpoint_ep{ep+1}.pt"), episode=ep + 1)

    csv_file.close()
    agent.save(str(ckpt_dir.parent / "pacman_dqn_final.pt"), episode=start_episode + episodes)
    env.close()
    print(f"Done. Best 50-episode average: {best_avg:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Pac-Man DQN")
    parser.add_argument("--config", default="configs/pacman_dqn.yaml")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    train_pacman(args.config, args.episodes, args.seed, args.resume)