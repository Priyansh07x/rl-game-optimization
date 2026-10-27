"""Flappy Bird Gymnasium environment wrapper matching PacManEnv conventions.

Underlying package: flappy-bird-gymnasium (v0.4.0)
Environment ID: FlappyBird-v0

Reward structure of underlying env (verified on v0.4.0):
    +0.1 per frame alive
    +1.0 per pipe passed
    -1.0 on crash (terminal)
"""

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import flappy_bird_gymnasium

# Verified reward values from flappy-bird-gymnasium package:
# +0.1 per frame alive, +1.0 per pipe passed, -1.0 on crash


class FlappyEnv:
    """Wrapper for Flappy Bird environment providing 84x84 grayscale image observations.

    Rendering details:
        When user specifies `render_mode="human"`, the underlying environment is initialized
        with `render_mode="human"` for Pygame display updates, and frames are extracted from the
        Pygame display surface. When `render_mode=None`, the underlying environment is initialized
        with `render_mode="rgb_array"` for headless RGB array rendering without displaying a window.
    """

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.underlying_render_mode = "human" if render_mode == "human" else "rgb_array"

        self.env = gym.make(
            "FlappyBird-v0",
            render_mode=self.underlying_render_mode,
            use_lidar=False,
        )

        assert self.env.action_space.n == 2, f"Expected 2 actions, got {self.env.action_space.n}"
        self.action_space = self.env.action_space
        self.num_actions = 2
        self.action_names = {0: "NOOP", 1: "FLAP"}

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(84, 84, 1),
            dtype=np.float32,
        )

        self.total_reward = 0.0
        self.episode_steps = 0
        self.max_episode_steps = 10000

    def _get_raw_frame(self):
        """Retrieve RGB array from underlying environment."""
        if self.underlying_render_mode == "human":
            import pygame

            self.env.render()
            surface = self.env.unwrapped._surface
            # Transpose (W, H, 3) -> (H, W, 3)
            return np.transpose(pygame.surfarray.array3d(surface), axes=(1, 0, 2))
        else:
            return self.env.render()

    def _preprocess_frame(self, frame):
        """Preprocess raw RGB frame into (84, 84, 1) float32 in [0, 1]."""
        if len(frame.shape) == 3:
            frame = np.mean(frame, axis=2).astype(np.uint8)

        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        frame = np.expand_dims(frame, axis=2)
        return frame.astype(np.float32) / 255.0

    def reset(self, seed=None):
        """Reset environment and return preprocessed observation and info."""
        _, info = self.env.reset(seed=seed)
        self.total_reward = 0.0
        self.episode_steps = 0

        raw_frame = self._get_raw_frame()
        obs = self._preprocess_frame(raw_frame)

        step_info = {
            **info,
            "score": info.get("score", 0),
            "episode_return": self.total_reward,
        }
        return obs, step_info

    def step(self, action):
        """Execute action (0=do nothing, 1=flap) and return 5-tuple."""
        _, reward, terminated, truncated, info = self.env.step(action)

        self.episode_steps += 1
        self.total_reward += reward

        if self.episode_steps >= self.max_episode_steps:
            truncated = True

        raw_frame = self._get_raw_frame()
        obs = self._preprocess_frame(raw_frame)

        step_info = {
            **info,
            "score": info.get("score", 0),
            "episode_return": self.total_reward,
        }
        return obs, reward, terminated, truncated, step_info

    def render(self):
        """Render frame."""
        if self.render_mode == "human":
            self.env.render()
            return None
        elif self.render_mode == "rgb_array":
            return self._get_raw_frame()
        return None

    def close(self):
        """Close underlying environment."""
        self.env.close()
