"""Snake Refined DQN Agent implementation."""

import copy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.agents.dqn.dual_replay import DualReplayBuffer
from src.agents.dqn.network import DuelingDQN
from src.common.config import load_config


class SnakeDQNAgent:
    """Refined Deep Q-Network Agent for Snake.

    Integrates DuelingDQN architecture and DualReplayBuffer following
    Official Milestone 3 specifications.
    """

    def __init__(
        self,
        config: Optional[Union[Dict[str, Any], str, Path]] = None,
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """Initialize SnakeDQNAgent.

        Args:
            config: Optional config dict or path to YAML config file.
            device: Computing device identifier ('auto', 'cpu', 'cuda').
            seed: Random seed for reproducibility.
        """
        # Load configuration
        if config is None:
            config = load_config("configs/snake_dqn.yaml")
        elif isinstance(config, (str, Path)):
            config = load_config(config)

        self.config = config

        # Device selection
        if device is None:
            device = config.get("device", "auto")

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Hyperparameters
        self.state_shape = tuple(config.get("state", {}).get("shape", [4, 64, 64]))
        self.num_actions = 4
        self.gamma = float(config.get("training", {}).get("gamma", 0.99))
        self.learning_rate = float(config.get("training", {}).get("learning_rate", 0.0001))
        self.batch_size = int(config.get("training", {}).get("batch_size", 32))
        self.target_update_steps = int(config.get("training", {}).get("target_update_steps", 500))
        self.training_gap = int(config.get("training", {}).get("training_gap", 4))

        # Epsilon decay schedule parameters
        self.epsilon_start = float(config.get("epsilon", {}).get("start", 1.0))
        self.epsilon_end = float(config.get("epsilon", {}).get("end", 0.02))
        self.decay_frames = int(config.get("epsilon", {}).get("decay_frames", 200000))

        # Replay buffer parameters
        replay_cfg = config.get("replay", {})
        high_cap = int(replay_cfg.get("high_error_capacity", 35000))
        low_cap = int(replay_cfg.get("low_error_capacity", 15000))
        td_thresh = float(replay_cfg.get("td_threshold", 0.5))
        high_frac = float(replay_cfg.get("high_error_fraction", 0.70))

        # Seed & RNG
        self.seed = seed if seed is not None else config.get("seed", 42)
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        # Initialize networks
        conv_channels = config.get("network", {}).get("conv_channels", [16, 32, 64])
        fc_units = config.get("network", {}).get("fully_connected_units", 256)

        self.online_net = DuelingDQN(
            in_channels=self.state_shape[0],
            num_actions=self.num_actions,
            conv_channels=conv_channels,
            fully_connected_units=fc_units,
            input_shape=(self.state_shape[1], self.state_shape[2]),
        ).to(self.device)

        self.target_net = DuelingDQN(
            in_channels=self.state_shape[0],
            num_actions=self.num_actions,
            conv_channels=conv_channels,
            fully_connected_units=fc_units,
            input_shape=(self.state_shape[1], self.state_shape[2]),
        ).to(self.device)

        self.update_target_network()
        for param in self.target_net.parameters():
            param.requires_grad = False

        # Optimizer
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.learning_rate)

        # Dual Replay Buffer
        self.replay_buffer = DualReplayBuffer(
            high_capacity=high_cap,
            low_capacity=low_cap,
            td_threshold=td_thresh,
            high_fraction=high_frac,
            seed=self.seed,
        )

        # Tracking variables
        self.frame_count = 0
        self.update_steps = 0

    def get_epsilon(self) -> float:
        """Calculate current epsilon value based on frame progression."""
        if self.frame_count >= self.decay_frames:
            return self.epsilon_end
        fraction = self.frame_count / float(self.decay_frames)
        eps = self.epsilon_start - fraction * (self.epsilon_start - self.epsilon_end)
        return float(max(self.epsilon_end, eps))

    @property
    def epsilon(self) -> float:
        """Current epsilon value property."""
        return self.get_epsilon()

    def update_target_network(self) -> None:
        """Synchronize target network weights from online network."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    def select_action(self, state: Any, eval_mode: bool = False) -> int:
        """Select an action using epsilon-greedy strategy.

        Args:
            state: State array/tensor of shape (4, 64, 64) or (1, 4, 64, 64).
            eval_mode: If True, exploration is disabled (epsilon = 0.0).

        Returns:
            Selected action integer in range [0, 3].
        """
        eps = 0.0 if eval_mode else self.get_epsilon()

        if not eval_mode and np.random.rand() < eps:
            return int(np.random.randint(0, self.num_actions))

        if not isinstance(state, torch.Tensor):
            tensor_state = torch.tensor(state, dtype=torch.float32, device=self.device)
        else:
            tensor_state = state.to(dtype=torch.float32, device=self.device)

        if tensor_state.ndim == 3:
            tensor_state = tensor_state.unsqueeze(0)

        with torch.no_grad():
            q_values = self.online_net(tensor_state)
            action = int(torch.argmax(q_values, dim=1).item())

        return action

    def store_transition(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool,
        td_error: Optional[float] = None,
    ) -> None:
        """Add transition to replay buffer.

        If td_error is None, default to 1.0 (high TD-error threshold default)
        so un-updated experiences route to high-TD error partition until trained on.
        """
        if td_error is None:
            td_error = 1.0

        self.replay_buffer.add(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            td_error=td_error,
        )

    def learn(self) -> Optional[Dict[str, float]]:
        """Perform a single DQN learning step using minibatch sampled from DualReplayBuffer.

        Returns:
            Dictionary of training statistics (loss, avg_q, mean_td_error), or None if insufficient data.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        self.update_steps += 1

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        b_states = torch.tensor(states, dtype=torch.float32, device=self.device)
        b_actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        b_rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        b_next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        b_dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Current Q-values from online network
        current_q = self.online_net(b_states).gather(1, b_actions)

        # Target Q-values from target network
        with torch.no_grad():
            max_next_q = self.target_net(b_next_states).max(dim=1, keepdim=True)[0]
            target_q = b_rewards + (1.0 - b_dones) * self.gamma * max_next_q

        # Calculate TD-errors for logging / priority routing
        td_errors = torch.abs(target_q - current_q).detach()

        # Smooth L1 / Huber loss
        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Periodic target network synchronization
        if self.update_steps % self.target_update_steps == 0:
            self.update_target_network()

        return {
            "loss": float(loss.item()),
            "avg_q": float(current_q.mean().item()),
            "mean_td_error": float(td_errors.mean().item()),
            "update_steps": self.update_steps,
        }

    def get_checkpoint_state(self) -> Dict[str, Any]:
        """Expose state dictionary structure for future checkpoint manager."""
        return {
            "online_net_state_dict": self.online_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "frame_count": self.frame_count,
            "update_steps": self.update_steps,
            "epsilon": self.get_epsilon(),
            "config": self.config,
        }

    def load_checkpoint_state(self, state: Dict[str, Any]) -> None:
        """Restore agent parameters, optimizer state, and training counters from checkpoint dictionary."""
        if "online_net_state_dict" in state:
            self.online_net.load_state_dict(state["online_net_state_dict"])
        if "target_net_state_dict" in state:
            self.target_net.load_state_dict(state["target_net_state_dict"])
        if "optimizer_state_dict" in state:
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
        if "frame_count" in state:
            self.frame_count = int(state["frame_count"])
        if "update_steps" in state:
            self.update_steps = int(state["update_steps"])

