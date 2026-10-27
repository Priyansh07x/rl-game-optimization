import numpy as np
from collections import Counter
from src.environments.pacman.environment import PacManEnv
from src.agents.dqn.pacman_dqn import PacManDQNAgent

env = PacManEnv()
agent = PacManDQNAgent(num_actions=env.action_space.n)
agent.load("checkpoints/pacman_dqn_best.pt")

obs, _ = env.reset(seed=0)
state = np.zeros((4, 84, 84), dtype=np.float32)
state[-1] = obs.squeeze()

actions, q_spreads = [], []
for _ in range(300):
    q = agent.online_net(torch.as_tensor(state).unsqueeze(0))[0].detach().numpy()
    actions.append(int(q.argmax()))
    q_spreads.append(float(q.max() - q.min()))
    obs, r, term, trunc, _ = env.step(actions[-1])
    state = np.roll(state, -1, axis=0)
    state[-1] = obs.squeeze()
    if term or trunc:
        break

print("Action counts:", Counter(actions))
print("Mean Q-spread:", np.mean(q_spreads))
print("Steps:", len(actions))
env.close()