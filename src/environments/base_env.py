"""Base RL environment interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import gymnasium as gym
import numpy as np


class BaseEnv(gym.Env, ABC):
    """Abstract base environment for RL project games."""

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state."""
        super().reset(seed=seed, options=options)
        return None, {}

    @abstractmethod
    def step(
        self, action: Any
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advance one environment step."""
        pass
