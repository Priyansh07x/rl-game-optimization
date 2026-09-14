"""Dueling DQN Agent module."""

from src.agents.dqn.network import DuelingDQN
from src.agents.dqn.dual_replay import DualReplayBuffer
from src.agents.dqn.agent import SnakeDQNAgent

__all__ = ["DuelingDQN", "DualReplayBuffer", "SnakeDQNAgent"]
