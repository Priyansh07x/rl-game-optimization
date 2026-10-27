"""General DQN evaluation entrypoint for Pac-Man, Snake, and Flappy Bird."""

import argparse
from pathlib import Path
import time
import numpy as np
import torch

from src.environments import make_env
from src.agents.dqn.pacman_dqn import PacManDQNAgent


def evaluate(game: str, model_path: str, episodes: int = 100, render: bool = False, seed: int = 1000):
    """Evaluate a trained DQN agent on a specified game.

    Args:
        game: Name of game ('pacman', 'snake', 'flappy').
        model_path: Path to PyTorch model checkpoint.
        episodes: Number of evaluation episodes.
        render: Whether to display rendered gameplay.
        seed: Initial evaluation seed (defaults to 1000).

    Returns:
        Tuple of (rewards, lengths).
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = make_env(game, render_mode="human" if render else None)
    agent = PacManDQNAgent(num_actions=env.num_actions, device=device)
    agent.load(str(model_path))
    agent.online_net.eval()
    print(f"Loaded model: {model_path} (trained steps: {agent.total_steps})")

    rewards, lengths = [], []
    t0 = time.time()
    for ep in range(episodes):
        obs, step_info = env.reset(seed=seed + ep if seed is not None else None)  # fixed, held-out seeds
        state = np.zeros((4, 84, 84), dtype=np.float32)
        state[-1] = obs.squeeze()

        ep_reward, ep_steps, done = 0.0, 0, False
        while not done:
            action = agent.select_action(state, greedy=True)
            obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated

            state = np.roll(state, -1, axis=0)
            state[-1] = obs.squeeze()
            ep_reward += reward
            ep_steps += 1

        rewards.append(ep_reward)
        lengths.append(ep_steps)
        reason = "crashed" if terminated else "max steps reached"
        score = step_info.get("score", 0)
        print(f"Episode {ep+1}/{episodes} | {reason} | score {score} | reward {ep_reward:.0f} | steps {ep_steps}")
        if render:
            time.sleep(0.6)

    env.close()
    r = np.array(rewards)
    print("\n" + "=" * 60)
    print(f"Episodes:        {episodes}")
    print(f"Mean reward:     {r.mean():.1f} \u00b1 {r.std():.1f}")
    print(f"Median reward:   {np.median(r):.1f}")
    print(f"Min / Max:       {r.min():.0f} / {r.max():.0f}")
    print(f"Mean length:     {np.mean(lengths):.1f} steps")
    print(f"Unique rewards:  {len(set(rewards))}")
    print(f"Eval time:       {(time.time()-t0)/60:.1f} min")
    print("=" * 60)
    return rewards, lengths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DQN for Atari & Grid Games")
    parser.add_argument("--game", choices=["pacman", "snake", "flappy"], default="pacman")
    parser.add_argument("--model", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    evaluate(args.game, args.model, args.episodes, args.render, args.seed)
