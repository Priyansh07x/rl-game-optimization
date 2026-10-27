import argparse
import time
from pathlib import Path

import numpy as np
import torch

from src.environments.pacman.environment import PacManEnv
from src.agents.dqn.pacman_dqn import PacManDQNAgent


def evaluate_pacman(model_path, episodes=100, render=False):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = PacManEnv(render_mode="human" if render else None)
    agent = PacManDQNAgent(num_actions=env.action_space.n, device=device)
    agent.load(str(model_path))
    agent.online_net.eval()
    print(f"Loaded model: {model_path} (trained steps: {agent.total_steps})")

    rewards, lengths = [], []
    t0 = time.time()
    for ep in range(episodes):
        obs, _ = env.reset(seed=1000 + ep)  # fixed, held-out seeds
        state = np.zeros((4, 84, 84), dtype=np.float32)
        state[-1] = obs.squeeze()

        ep_reward, ep_steps, done = 0.0, 0, False
        while not done:
            action = agent.select_action(state, greedy=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = np.roll(state, -1, axis=0)
            state[-1] = obs.squeeze()
            ep_reward += reward
            ep_steps += 1

        rewards.append(ep_reward)
        lengths.append(ep_steps)
        print(f"Episode {ep+1}/{episodes} | reward {ep_reward:.0f} | steps {ep_steps}")

    env.close()
    r = np.array(rewards)
    print("\n" + "=" * 60)
    print(f"Episodes:        {episodes}")
    print(f"Mean reward:     {r.mean():.1f} ± {r.std():.1f}")
    print(f"Median reward:   {np.median(r):.1f}")
    print(f"Min / Max:       {r.min():.0f} / {r.max():.0f}")
    print(f"Mean length:     {np.mean(lengths):.1f} steps")
    print(f"Unique rewards:  {len(set(rewards))}")
    print(f"Eval time:       {(time.time()-t0)/60:.1f} min")
    print("=" * 60)
    return rewards, lengths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Pac-Man DQN")
    parser.add_argument("--model", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    evaluate_pacman(args.model, args.episodes, args.render)