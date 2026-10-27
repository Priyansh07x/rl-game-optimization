"""Snake Gymnasium environment matching PacManEnv conventions.

Constants:
    SURVIVAL_REWARD: Dense step reward (+0.01) given on non-terminal surviving steps.
    APPLE_REWARD: Reward (+1.0) awarded when an apple is eaten.
    COLLISION_REWARD: Penalty (-1.0) applied upon wall or self collision.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Dense per-step survival reward to guide DQN early in training
SURVIVAL_REWARD = 0.01
APPLE_REWARD = 1.0
COLLISION_REWARD = -1.0


class SnakeEnv:
    """Pure-Python 2D Snake environment with 84x84 grayscale observation space."""

    def __init__(self, render_mode=None, grid_size=12, cell_px=20, seed=None):
        self.render_mode = render_mode
        self.grid_size = grid_size
        self.cell_px = cell_px
        self.max_episode_steps = 5000

        # Action space: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        self.action_space = spaces.Discrete(4)
        self.num_actions = 4
        self.action_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
        self.opposite_actions = {0: 1, 1: 0, 2: 3, 3: 2}
        self.action_vectors = {
            0: (0, -1),  # UP
            1: (0, 1),   # DOWN
            2: (-1, 0),  # LEFT
            3: (1, 0),   # RIGHT
        }

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(84, 84, 1),
            dtype=np.float32,
        )

        self.rng = np.random.default_rng(seed)
        self.snake = []
        self.direction = 3  # Start facing RIGHT
        self.apple = None
        self.score = 0
        self.total_reward = 0.0
        self.episode_steps = 0

        # Pygame display handles (only initialized if render_mode == 'human')
        self.window = None
        self.clock = None

    def reset(self, seed=None):
        """Reset environment to initial state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.direction = 3  # Start facing RIGHT
        cy = self.grid_size // 2
        # Initial snake length 3 facing RIGHT
        self.snake = [(3, cy), (2, cy), (1, cy)]
        self.score = 0
        self.total_reward = 0.0
        self.episode_steps = 0

        self._spawn_apple()

        rgb_frame = self._render_rgb_grid()
        obs = self._preprocess_frame(rgb_frame)

        if self.render_mode == "human":
            self._render_human(rgb_frame)

        info = {
            "score": self.score,
            "length": len(self.snake),
            "episode_return": self.total_reward,
        }
        return obs, info

    def step(self, action):
        """Step the environment with chosen action."""
        action = int(action)
        # Reject 180-degree reversal by ignoring the action
        if action == self.opposite_actions.get(self.direction):
            action = self.direction

        self.direction = action
        dx, dy = self.action_vectors[self.direction]
        head_x, head_y = self.snake[0]
        new_head = (head_x + dx, head_y + dy)

        self.episode_steps += 1
        terminated = False
        truncated = False

        # Collision check: wall collision
        wall_collision = not (0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size)
        # Collision check: self collision
        self_collision = new_head in self.snake[:-1]

        if wall_collision or self_collision:
            terminated = True
            reward = COLLISION_REWARD
        else:
            self.snake.insert(0, new_head)
            if new_head == self.apple:
                self.score += 1
                reward = APPLE_REWARD
                self._spawn_apple()
            else:
                self.snake.pop()
                reward = SURVIVAL_REWARD

        self.total_reward += reward

        if self.episode_steps >= self.max_episode_steps:
            truncated = True

        rgb_frame = self._render_rgb_grid()
        obs = self._preprocess_frame(rgb_frame)

        if self.render_mode == "human":
            self._render_human(rgb_frame)

        info = {
            "score": self.score,
            "length": len(self.snake),
            "episode_return": self.total_reward,
        }
        return obs, reward, terminated, truncated, info

    def _spawn_apple(self):
        """Spawn an apple on an empty cell uniformly at random."""
        snake_set = set(self.snake)
        empty_cells = [
            (x, y)
            for x in range(self.grid_size)
            for y in range(self.grid_size)
            if (x, y) not in snake_set
        ]
        if empty_cells:
            idx = int(self.rng.integers(0, len(empty_cells)))
            self.apple = empty_cells[idx]
        else:
            self.apple = None

    def _render_rgb_grid(self):
        """Build full resolution RGB numpy frame of the grid."""
        h = self.grid_size * self.cell_px
        w = self.grid_size * self.cell_px
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Draw apple
        if self.apple is not None:
            ax, ay = self.apple
            frame[
                ay * self.cell_px : (ay + 1) * self.cell_px,
                ax * self.cell_px : (ax + 1) * self.cell_px,
            ] = [255, 50, 50]

        # Draw snake body
        for bx, by in self.snake[1:]:
            frame[
                by * self.cell_px : (by + 1) * self.cell_px,
                bx * self.cell_px : (bx + 1) * self.cell_px,
            ] = [0, 180, 0]

        # Draw snake head
        if self.snake:
            hx, hy = self.snake[0]
            frame[
                hy * self.cell_px : (hy + 1) * self.cell_px,
                hx * self.cell_px : (hx + 1) * self.cell_px,
            ] = [150, 255, 150]

        return frame

    def _preprocess_frame(self, frame):
        """Preprocess raw RGB frame to (84, 84, 1) normalized float32."""
        if len(frame.shape) == 3:
            frame = np.mean(frame, axis=2).astype(np.uint8)

        from cv2 import resize, INTER_AREA
        frame = resize(frame, (84, 84), interpolation=INTER_AREA)
        frame = np.expand_dims(frame, axis=2)
        return frame.astype(np.float32) / 255.0

    def render(self):
        """Render current frame based on render_mode."""
        if self.render_mode == "human":
            rgb_frame = self._render_rgb_grid()
            self._render_human(rgb_frame)
            return None
        elif self.render_mode == "rgb_array":
            return self._render_rgb_grid()
        return None

    def _render_human(self, rgb_frame):
        """Blit grid frame to Pygame window."""
        import pygame

        h, w, _ = rgb_frame.shape
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((w, h))
            pygame.display.set_caption("Snake")
            self.clock = pygame.time.Clock()

        # Pygame surfarray expects (W, H, 3)
        surface = pygame.surfarray.make_surface(np.transpose(rgb_frame, (1, 0, 2)))
        self.window.blit(surface, (0, 0))
        pygame.event.pump()
        pygame.display.flip()
        if self.clock is not None:
            self.clock.tick(15)

    def close(self):
        """Release any opened Pygame window."""
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
