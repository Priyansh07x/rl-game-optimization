"""Base Abstract Tabular RL Agent.

Provides common Q-table storage, state/action validation, epsilon-greedy action selection,
epsilon decay scheduling, and state dict serialization for tabular agents.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np


class BaseTabularAgent(ABC):
    """Abstract base class for tabular reinforcement learning agents.

    Attributes:
        num_states: Total number of discrete states in environment (default 75).
        num_actions: Total number of discrete actions in environment (default 2).
        alpha: Learning rate parameter (default 0.1).
        gamma: Discount factor parameter (default 0.99).
        epsilon_start: Initial exploration rate (default 1.0).
        epsilon_end: Minimum exploration rate (default 0.05).
        decay_episodes: Episode horizon over which epsilon decays (default 5000).
        seed: Optional random seed for reproducible exploration.
        q_table: 2D NumPy float64 matrix of shape (num_states, num_actions).
        epsilon: Current exploration rate.
        episode: Current training episode count.
    """

    def __init__(
        self,
        num_states: int = 75,
        num_actions: int = 2,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        decay_episodes: int = 5000,
        seed: Optional[int] = None,
    ):
        """Initialize BaseTabularAgent.

        Args:
            num_states: Number of discrete states (default 75).
            num_actions: Number of discrete actions (default 2).
            alpha: Learning rate.
            gamma: Discount factor.
            epsilon_start: Initial epsilon value (e.g. 1.0).
            epsilon_end: Minimum epsilon value (e.g. 0.05).
            decay_episodes: Number of episodes over which epsilon decays.
            seed: Random seed for action sampling reproducibility.
        """
        if num_states <= 0:
            raise ValueError(f"num_states must be > 0, got {num_states}")
        if num_actions <= 0:
            raise ValueError(f"num_actions must be > 0, got {num_actions}")

        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.decay_episodes = int(decay_episodes)
        self.seed = seed

        # Initialize Q-table matrix with float64 zeros
        self.q_table: np.ndarray = np.zeros((self.num_states, self.num_actions), dtype=np.float64)

        # Internal tracking
        self.epsilon: float = self.epsilon_start
        self.episode: int = 0

        # Random Number Generator for reproducible epsilon-greedy exploration
        self.rng = np.random.default_rng(self.seed)

    def validate_state(self, state: int) -> int:
        """Validate that state integer index is within legal bounds [0, num_states).

        Args:
            state: State integer index.

        Returns:
            Validated integer state index.

        Raises:
            ValueError: If state is not integer or out of bounds.
        """
        if not isinstance(state, (int, np.integer)):
            raise ValueError(f"State index must be an integer, got {type(state)}")
        state_idx = int(state)
        if not (0 <= state_idx < self.num_states):
            raise ValueError(f"State index {state_idx} out of valid bounds [0, {self.num_states})")
        return state_idx

    def validate_action(self, action: int) -> int:
        """Validate that action integer index is within legal bounds [0, num_actions).

        Args:
            action: Action integer index.

        Returns:
            Validated integer action index.

        Raises:
            ValueError: If action is not integer or out of bounds.
        """
        if not isinstance(action, (int, np.integer)):
            raise ValueError(f"Action index must be an integer, got {type(action)}")
        action_idx = int(action)
        if not (0 <= action_idx < self.num_actions):
            raise ValueError(f"Action index {action_idx} out of valid bounds [0, {self.num_actions})")
        return action_idx

    def select_greedy_action(self, state: int) -> int:
        """Select the action that maximizes Q(state, a).

        Tie-breaking rule:
            When multiple actions achieve the exact same maximum Q-value, tie-breaking
            is deterministic by selecting the smallest action index (via np.argmax).

        Args:
            state: Discrete state index in [0, num_states).

        Returns:
            Best action index in [0, num_actions).
        """
        st = self.validate_state(state)
        q_vals = self.q_table[st]
        # np.argmax returns the first occurrence of the maximum value (smallest action index)
        return int(np.argmax(q_vals))

    def select_action(self, state: int, eval_mode: bool = False) -> int:
        """Select an action using epsilon-greedy policy.

        Args:
            state: Discrete state index in [0, num_states).
            eval_mode: If True, uses purely greedy selection (effective epsilon = 0.0)
                       without modifying self.epsilon.

        Returns:
            Action index in [0, num_actions).
        """
        st = self.validate_state(state)
        effective_eps = 0.0 if eval_mode else self.epsilon

        if self.rng.random() < effective_eps:
            # Explore: uniform random action selection
            return int(self.rng.integers(0, self.num_actions))
        else:
            # Exploit: greedy action selection
            return self.select_greedy_action(st)

    def decay_epsilon(self, episode: Optional[int] = None) -> float:
        """Update self.epsilon following a deterministic linear decay schedule.

        Schedule definition:
            For episode ep in [0, decay_episodes]:
                progress = min(1.0, ep / decay_episodes)
                epsilon = max(epsilon_end, epsilon_start - progress * (epsilon_start - epsilon_end))

        Args:
            episode: Optional explicit 1-indexed or 0-indexed episode number. If None, increments
                     self.episode by 1 and uses self.episode.

        Returns:
            Updated float epsilon value.
        """
        if episode is not None:
            self.episode = int(episode)
        else:
            self.episode += 1

        if self.decay_episodes <= 0:
            self.epsilon = self.epsilon_end
            return self.epsilon

        progress = min(1.0, max(0.0, float(self.episode) / float(self.decay_episodes)))
        decayed = self.epsilon_start - progress * (self.epsilon_start - self.epsilon_end)
        self.epsilon = max(self.epsilon_end, float(decayed))
        return self.epsilon

    @property
    def q_table_size(self) -> int:
        """Return total number of entries in the Q-table (75 x 2 = 150)."""
        return int(self.q_table.size)

    @property
    def non_zero_entries(self) -> int:
        """Return total count of non-zero entries in the Q-table."""
        return int(np.count_nonzero(self.q_table))

    def get_checkpoint_state(self) -> Dict[str, Any]:
        """Serialize complete agent state for checkpointing.

        Returns:
            Dict containing Q-table, episode count, epsilon, hyperparameters, and RNG state.
        """
        return {
            "q_table": self.q_table.copy(),
            "episode": int(self.episode),
            "epsilon": float(self.epsilon),
            "num_states": int(self.num_states),
            "num_actions": int(self.num_actions),
            "alpha": float(self.alpha),
            "gamma": float(self.gamma),
            "epsilon_start": float(self.epsilon_start),
            "epsilon_end": float(self.epsilon_end),
            "decay_episodes": int(self.decay_episodes),
            "seed": self.seed,
            "rng_state": self.rng.bit_generator.state,
        }

    def load_checkpoint_state(self, state_dict: Dict[str, Any]) -> None:
        """Restore agent state from checkpoint dict.

        Args:
            state_dict: Dict containing saved checkpoint entries.
        """
        if "q_table" not in state_dict:
            raise ValueError("Invalid checkpoint dict: missing 'q_table'")

        q_table = np.array(state_dict["q_table"], dtype=np.float64)
        if q_table.shape != (self.num_states, self.num_actions):
            raise ValueError(
                f"Checkpoint Q-table shape {q_table.shape} does not match agent shape {(self.num_states, self.num_actions)}"
            )

        self.q_table = q_table
        self.episode = int(state_dict.get("episode", self.episode))
        self.epsilon = float(state_dict.get("epsilon", self.epsilon))
        self.alpha = float(state_dict.get("alpha", self.alpha))
        self.gamma = float(state_dict.get("gamma", self.gamma))

        if "rng_state" in state_dict:
            try:
                self.rng.bit_generator.state = state_dict["rng_state"]
            except Exception:
                pass

    @abstractmethod
    def update(self, *args: Any, **kwargs: Any) -> float:
        """Abstract update method to be implemented by QLearningAgent and SARSAAgent."""
        pass
