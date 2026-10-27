import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque


class PacManDQN(nn.Module):
    """Dueling DQN. Expects float32 states in [0, 1] with shape (B, 4, 84, 84)."""

    def __init__(self, input_channels=4, num_actions=9):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4), nn.ReLU(),  # 84 -> 20
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),              # 20 -> 9
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),              # 9 -> 7
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(64 * 7 * 7, 512), nn.ReLU())
        self.value = nn.Linear(512, 1)
        self.advantage = nn.Linear(512, num_actions)

    def forward(self, x):
        h = self.fc(self.conv(x))
        v = self.value(h)
        a = self.advantage(h)
        return v + a - a.mean(dim=1, keepdim=True)


class ReplayBuffer:
    """Stores frames as uint8 to keep memory manageable."""

    def __init__(self, capacity=20_000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((
            np.round(state * 255).astype(np.uint8),
            action, reward,
            np.round(next_state * 255).astype(np.uint8),
            done,
        ))

    def sample(self, batch_size):
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        s, a, r, s2, d = zip(*(self.buffer[i] for i in idx))
        return (np.stack(s), np.array(a), np.array(r, dtype=np.float32),
                np.stack(s2), np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


class PacManDQNAgent:
    def __init__(self, num_actions=9, lr=1e-4, gamma=0.99,
                 buffer_capacity=20_000, batch_size=32, learn_start=5_000,
                 train_every=4, target_update=1_000,
                 eps_start=1.0, eps_end=0.01, eps_decay_steps=100_000,
                 device="cpu"):
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.learn_start = learn_start
        self.train_every = train_every
        self.target_update = target_update
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.device = torch.device(device)

        self.online_net = PacManDQN(4, num_actions).to(self.device)
        self.target_net = PacManDQN(4, num_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)
        self.total_steps = 0

    @property
    def epsilon(self):
        frac = min(1.0, self.total_steps / self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def select_action(self, state, greedy=False):
        if not greedy and np.random.random() < self.epsilon:
            return int(np.random.randint(self.num_actions))
        x = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return int(self.online_net(x).argmax(dim=1).item())

    def update(self, state, action, reward, next_state, done):
        """Store one transition, maybe train. Returns the loss, or None if no update ran."""
        # Clip rewards for training stability. Logged episode rewards stay raw.
        self.buffer.add(state, action, float(np.clip(reward, -1.0, 1.0)), next_state, done)
        self.total_steps += 1

        loss = None
        if len(self.buffer) >= self.learn_start and self.total_steps % self.train_every == 0:
            loss = self._train_step()
        if self.total_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        return loss

    def _train_step(self):
        s, a, r, s2, d = self.buffer.sample(self.batch_size)
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device) / 255.0
        s2 = torch.as_tensor(s2, dtype=torch.float32, device=self.device) / 255.0
        a = torch.as_tensor(a, dtype=torch.long, device=self.device)
        r = torch.as_tensor(r, device=self.device)
        d = torch.as_tensor(d, device=self.device)

        q = self.online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            target = r + self.gamma * (1.0 - d) * self.target_net(s2).max(dim=1).values

        loss = F.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()
        return loss.item()

    def save(self, path, episode=0):
        torch.save({
            "model_state": self.online_net.state_dict(),
            "target_state": self.target_net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "episode": episode,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["model_state"])
        self.target_net.load_state_dict(ckpt["target_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.total_steps = ckpt["total_steps"]
        return ckpt.get("episode", 0)