"""Flappy Bird Gymnasium Environment Wrapper.

Wraps flappy-bird-gymnasium's FlappyBird-v0 environment to conform to project standards:
- Explicit action mapping:
    Project action 0 = FLAP -> Underlying action 1
    Project action 1 = NO FLAP -> Underlying action 0
- Project reward structure:
    Survival: +0.5
    Pipe Pass: +5.0
    Collision: -1000.0
- Integrates FlappyDiscretizer for 75-state tabular RL compatibility.
- Exposes standard Gymnasium reset() / step() contract and helper get_state().
"""

from enum import IntEnum
from typing import Any, Dict, Optional, Tuple, Union
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import flappy_bird_gymnasium

from src.environments.base_env import BaseEnv
from src.preprocessing.flappy_preprocess import FlappyDiscretizer


class FlappyAction(IntEnum):
    """Explicit project-level action mapping for Flappy Bird."""
    FLAP = 0
    NO_FLAP = 1


# Translation from project action to underlying FlappyBird-v0 action API
# Underlying FlappyBird-v0: 0 = IDLE/NO_FLAP, 1 = FLAP
PROJECT_TO_NATIVE_ACTION: Dict[FlappyAction, int] = {
    FlappyAction.FLAP: 1,
    FlappyAction.NO_FLAP: 0,
}


class FlappyEnv(BaseEnv):
    """Project-level Flappy Bird environment wrapper."""

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    def __init__(
        self,
        survival_reward: float = 0.5,
        pipe_pass_reward: float = 5.0,
        collision_reward: float = -1000.0,
        render_mode: Optional[str] = None,
        discretizer: Optional[FlappyDiscretizer] = None,
        use_lidar: bool = False,
    ):
        """Initialize the Flappy Bird environment wrapper.

        Args:
            survival_reward: Reward received per survival step (default +0.5).
            pipe_pass_reward: Reward received when successfully passing a pipe (default +5.0).
            collision_reward: Reward received upon crash/termination (default -1000.0).
            render_mode: Rendering mode ("human", "rgb_array", or None).
            discretizer: Optional custom FlappyDiscretizer instance.
            use_lidar: Whether to use LiDAR (must be False for 12-element obs).
        """
        super().__init__()

        self.survival_reward = survival_reward
        self.pipe_pass_reward = pipe_pass_reward
        self.collision_reward = collision_reward
        self.render_mode = render_mode
        self.use_lidar = use_lidar

        # Initialize underlying Gymnasium environment
        self._env = gym.make("FlappyBird-v0", use_lidar=self.use_lidar, render_mode=self.render_mode)

        # Action space: Discrete(2) -> 0=FLAP, 1=NO_FLAP
        self.action_space = spaces.Discrete(2)

        # Observation space: 12 continuous features
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(12,),
            dtype=np.float64,
        )

        # Discretizer instance
        self.discretizer = discretizer if discretizer is not None else FlappyDiscretizer()

        # State tracking
        self.last_obs: Optional[np.ndarray] = None
        self.prev_score: int = 0
        self.steps: int = 0
        self.total_reward: float = 0.0
        self.terminated: bool = False
        self.truncated: bool = False

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment.

        Args:
            seed: Random seed for reproducibility.
            options: Additional Gymnasium reset options.

        Returns:
            Tuple of (raw_observation, info_dict).
        """
        super().reset(seed=seed)

        raw_obs, info = self._env.reset(seed=seed, options=options)
        self.last_obs = raw_obs
        self.prev_score = info.get("score", 0)
        self.steps = 0
        self.total_reward = 0.0
        self.terminated = False
        self.truncated = False

        # Add state extraction information to info
        state_idx, bin_tuple, continuous_feats = self.discretizer.process_obs(raw_obs)
        info.update({
            "state_index": state_idx,
            "x_bin": bin_tuple[0],
            "y_bin": bin_tuple[1],
            "v_bin": bin_tuple[2],
            "dx": continuous_feats[0],
            "dy": continuous_feats[1],
            "v_y": continuous_feats[2],
            "score": self.prev_score,
            "steps": self.steps,
        })

        return raw_obs, info

    def step(
        self, action: Union[int, FlappyAction]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment transition step with project action and reward mapping.

        Args:
            action: Project action (0=FLAP, 1=NO_FLAP).

        Returns:
            Tuple of (raw_obs, reward, terminated, truncated, info).
        """
        if self.terminated or self.truncated:
            raise RuntimeError("Cannot step on terminated/truncated environment. Call reset().")

        self.steps += 1
        proj_act = FlappyAction(int(action))

        # Map project action to native underlying action
        native_act = PROJECT_TO_NATIVE_ACTION[proj_act]

        # Execute step in underlying env
        raw_obs, native_reward, terminated, truncated, native_info = self._env.step(native_act)

        self.last_obs = raw_obs
        self.terminated = terminated
        self.truncated = truncated

        current_score = native_info.get("score", 0)
        pipe_passed = current_score > self.prev_score

        # Apply project reward specification
        if pipe_passed:
            reward = self.pipe_pass_reward
        elif terminated:
            reward = self.collision_reward
        else:
            reward = self.survival_reward

        self.prev_score = current_score
        self.total_reward += reward

        # State discretization info
        state_idx, bin_tuple, continuous_feats = self.discretizer.process_obs(raw_obs)

        info = {
            "score": current_score,
            "pipe_passed": pipe_passed,
            "steps": self.steps,
            "project_action": int(proj_act),
            "project_action_name": proj_act.name,
            "native_action": native_act,
            "state_index": state_idx,
            "x_bin": bin_tuple[0],
            "y_bin": bin_tuple[1],
            "v_bin": bin_tuple[2],
            "dx": continuous_feats[0],
            "dy": continuous_feats[1],
            "v_y": continuous_feats[2],
            "native_reward": native_reward,
        }

        return raw_obs, reward, terminated, truncated, info

    def get_state(self, obs: Optional[np.ndarray] = None) -> int:
        """Helper method to return the current discrete 75-state index.

        Args:
            obs: Optional observation vector. If None, uses last_obs.

        Returns:
            Integer state index in [0, 74].
        """
        target_obs = obs if obs is not None else self.last_obs
        if target_obs is None:
            raise ValueError("No observation available. Call reset() first.")
        state_idx, _, _ = self.discretizer.process_obs(target_obs)
        return state_idx

    def render(self) -> Optional[np.ndarray]:
        """Render frame."""
        return self._env.render()

    def close(self) -> None:
        """Close underlying environment."""
        self._env.close()
