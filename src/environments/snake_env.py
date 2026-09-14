"""Snake Gymnasium Environment.

Environment specification:
- Game map: 240 x 240 pixels
- Logical grid: 12 x 12
- Initial snake length: 3
- Initial direction: RIGHT
- Exactly one apple at a time
- Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
- Wall collision: terminates episode
- Self collision: terminates episode
- Reward: configurable (+1.0 apple, -1.0 collision, 0.0 step by default)
- Standard Gymnasium reset() / step() contract
"""

from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple, Union
from gymnasium import spaces
import numpy as np

from src.environments.base_env import BaseEnv


class SnakeAction(IntEnum):
    """Explicit, centralized action mapping for Snake."""
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


# Coordinate offsets (dx, dy) for each action
# Origin (0,0) is top-left: x is horizontal (0..grid_size-1), y is vertical (0..grid_size-1)
ACTION_DIRECTIONS: Dict[SnakeAction, Tuple[int, int]] = {
    SnakeAction.UP: (0, -1),
    SnakeAction.DOWN: (0, 1),
    SnakeAction.LEFT: (-1, 0),
    SnakeAction.RIGHT: (1, 0),
}

# Opposites mapping to prevent instant 180-degree self-collision
OPPOSITE_ACTIONS: Dict[SnakeAction, SnakeAction] = {
    SnakeAction.UP: SnakeAction.DOWN,
    SnakeAction.DOWN: SnakeAction.UP,
    SnakeAction.LEFT: SnakeAction.RIGHT,
    SnakeAction.RIGHT: SnakeAction.LEFT,
}


class SnakeEnv(BaseEnv):
    """Gymnasium-compliant Snake Environment.

    Follows all specifications from the project guide:
    - 240x240 pixel map / 12x12 logical grid
    - Explicit actions: UP(0), DOWN(1), LEFT(2), RIGHT(3)
    - Length starts at 3, initial direction RIGHT
    - Single apple spawned at unoccupied cell
    - Configurable reward structure
    - gymnasium reset() -> (obs, info), step() -> (obs, reward, terminated, truncated, info)
    """

    metadata = {"render_modes": ["rgb_array", "ansi", "human"], "render_fps": 10}

    def __init__(
        self,
        grid_size: int = 12,
        map_size: int = 240,
        initial_length: int = 3,
        initial_direction: SnakeAction = SnakeAction.RIGHT,
        apple_reward: float = 1.0,
        collision_reward: float = -1.0,
        step_reward: float = 0.0,
        max_steps: Optional[int] = None,
        obs_type: str = "grid",
        render_mode: Optional[str] = None,
        allow_reverse: bool = False,
        random_spawn: bool = False,
    ):
        """Initialize the Snake environment.

        Args:
            grid_size: Dimension of the logical grid (default: 12x12).
            map_size: Pixel dimensions for visual rendering (default: 240x240).
            initial_length: Initial length of snake body (default: 3).
            initial_direction: Initial movement direction (default: SnakeAction.RIGHT).
            apple_reward: Reward received when eating an apple (default: +1.0).
            collision_reward: Reward received upon collision/death (default: -1.0).
            step_reward: Reward received per survival step (default: 0.0).
            max_steps: Optional episode step limit for truncation.
            obs_type: Observation format - "grid" ((12, 12) int8) or "rgb" ((240, 240, 3) uint8).
            render_mode: Render mode ("rgb_array", "ansi", "human", or None).
            allow_reverse: Whether 180-degree direction reversals into neck are allowed.
            random_spawn: Whether to randomize initial snake placement on reset.
        """
        super().__init__()

        self.grid_size = grid_size
        self.map_size = map_size
        self.cell_size = map_size // grid_size
        self.initial_length = initial_length
        self.initial_direction = initial_direction
        self.apple_reward = apple_reward
        self.collision_reward = collision_reward
        self.step_reward = step_reward
        self.max_steps = max_steps
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.allow_reverse = allow_reverse
        self.random_spawn = random_spawn

        # Action space: Discrete(4) -> UP(0), DOWN(1), LEFT(2), RIGHT(3)
        self.action_space = spaces.Discrete(4)

        # Observation space definition
        if self.obs_type == "rgb":
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(self.map_size, self.map_size, 3),
                dtype=np.uint8,
            )
        elif self.obs_type == "grid":
            # Grid values: 0=Empty, 1=Snake Head, 2=Snake Body, 3=Apple
            self.observation_space = spaces.Box(
                low=0,
                high=3,
                shape=(self.grid_size, self.grid_size),
                dtype=np.int8,
            )
        else:
            raise ValueError(f"Unknown obs_type: {obs_type}. Must be 'grid' or 'rgb'.")

        # Internal state variables
        self.snake: List[Tuple[int, int]] = []
        self.direction: SnakeAction = self.initial_direction
        self.apple: Optional[Tuple[int, int]] = None
        self.score: int = 0
        self.steps: int = 0
        self.terminated: bool = False
        self.truncated: bool = False

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility.
            options: Additional options dictionary.

        Returns:
            Tuple of (initial_observation, info_dict).
        """
        super().reset(seed=seed)

        self.direction = self.initial_direction
        self.score = 0
        self.steps = 0
        self.terminated = False
        self.truncated = False

        # Deploy initial snake
        if self.random_spawn:
            # Pick a valid head position allowing body extension to the left
            head_x = self.np_random.integers(self.initial_length - 1, self.grid_size)
            head_y = self.np_random.integers(0, self.grid_size)
            self.snake = [(head_x - i, head_y) for i in range(self.initial_length)]
        else:
            # Deterministic default spawn centered vertically, facing right
            # Head at (initial_length + 1, mid_y), e.g. (4, 5) for 12x12 grid with length 3
            mid_y = self.grid_size // 2
            start_x = self.initial_length + 1
            self.snake = [(start_x - i, mid_y) for i in range(self.initial_length)]

        # Spawn the first apple
        self._spawn_apple()

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(
        self, action: Union[int, SnakeAction]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment transition step.

        Args:
            action: Action integer or SnakeAction (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT).

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        if self.terminated or self.truncated:
            raise RuntimeError("Cannot call step() on a terminated/truncated environment. Please call reset().")

        self.steps += 1
        act = SnakeAction(int(action))

        # Handle reverse direction input
        if not self.allow_reverse and len(self.snake) > 1:
            if act == OPPOSITE_ACTIONS[self.direction]:
                # Ignore reverse action, maintain current direction
                act = self.direction

        self.direction = act
        dx, dy = ACTION_DIRECTIONS[self.direction]

        head_x, head_y = self.snake[0]
        new_head = (head_x + dx, head_y + dy)

        # 1. Check Wall Collision
        if not (0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size):
            self.terminated = True
            reward = self.collision_reward
            info = self._get_info(collision="wall_collision")
            obs = self._get_obs()
            if self.render_mode == "human":
                self.render()
            return obs, reward, self.terminated, self.truncated, info

        # 2. Check Self Collision
        # Note: If not eating apple, the tail will vacate its cell at this step,
        # so colliding with self.snake[:-1] is self-collision.
        if new_head in self.snake[:-1]:
            self.terminated = True
            reward = self.collision_reward
            info = self._get_info(collision="self_collision")
            obs = self._get_obs()
            if self.render_mode == "human":
                self.render()
            return obs, reward, self.terminated, self.truncated, info

        # 3. Advance Snake Movement
        self.snake.insert(0, new_head)
        apple_eaten = (new_head == self.apple)

        if apple_eaten:
            self.score += 1
            reward = self.apple_reward

            # Check if grid is completely filled (game won)
            if len(self.snake) == self.grid_size * self.grid_size:
                self.terminated = True
                self.apple = None
            else:
                self._spawn_apple()
        else:
            # Remove tail if no apple was eaten
            self.snake.pop()
            reward = self.step_reward

        # Check step truncation
        if self.max_steps is not None and self.steps >= self.max_steps and not self.terminated:
            self.truncated = True

        obs = self._get_obs()
        info = self._get_info(apple_eaten=apple_eaten)

        if self.render_mode == "human":
            self.render()

        return obs, reward, self.terminated, self.truncated, info

    def _spawn_apple(self) -> None:
        """Spawn an apple at a random unoccupied grid position."""
        occupied = set(self.snake)
        empty_cells = [
            (x, y)
            for x in range(self.grid_size)
            for y in range(self.grid_size)
            if (x, y) not in occupied
        ]

        if not empty_cells:
            self.apple = None
            return

        idx = self.np_random.integers(0, len(empty_cells))
        self.apple = empty_cells[idx]

    def _get_obs(self) -> np.ndarray:
        """Construct the observation based on configured obs_type."""
        if self.obs_type == "grid":
            grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
            # Mark snake body
            for seg in self.snake[1:]:
                grid[seg[1], seg[0]] = 2
            # Mark snake head
            if self.snake:
                grid[self.snake[0][1], self.snake[0][0]] = 1
            # Mark apple
            if self.apple is not None:
                grid[self.apple[1], self.apple[0]] = 3
            return grid

        elif self.obs_type == "rgb":
            return self._render_rgb()

        raise ValueError(f"Invalid obs_type: {self.obs_type}")

    def _get_info(
        self,
        apple_eaten: bool = False,
        collision: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Construct the standard info dictionary."""
        return {
            "score": self.score,
            "snake_length": len(self.snake),
            "snake_head": self.snake[0] if self.snake else None,
            "snake_body": list(self.snake),
            "apple": self.apple,
            "direction": self.direction.name,
            "direction_value": int(self.direction),
            "steps": self.steps,
            "apple_eaten": apple_eaten,
            "collision": collision,
            "grid_size": self.grid_size,
            "map_size": self.map_size,
        }

    def _render_rgb(self) -> np.ndarray:
        """Render the 240x240 RGB pixel map.

        Color scheme:
        - Background: Dark background [15, 15, 15]
        - Snake Head: Bright green [0, 255, 64]
        - Snake Body: Medium green [0, 180, 48]
        - Apple: Bright red [255, 32, 32]
        """
        img = np.full((self.map_size, self.map_size, 3), 15, dtype=np.uint8)
        cs = self.cell_size

        # Draw apple
        if self.apple is not None:
            ax, ay = self.apple
            img[ay * cs : (ay + 1) * cs, ax * cs : (ax + 1) * cs] = [255, 32, 32]

        # Draw snake body
        for seg_x, seg_y in self.snake[1:]:
            img[seg_y * cs : (seg_y + 1) * cs, seg_x * cs : (seg_x + 1) * cs] = [0, 180, 48]

        # Draw snake head
        if self.snake:
            hx, hy = self.snake[0]
            img[hy * cs : (hy + 1) * cs, hx * cs : (hx + 1) * cs] = [0, 255, 64]

        return img

    def render(self) -> Optional[Union[np.ndarray, str]]:
        """Render the environment according to render_mode."""
        if self.render_mode is None:
            return None

        if self.render_mode == "rgb_array":
            return self._render_rgb()

        if self.render_mode in ("ansi", "human"):
            grid_chars = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
            # Body
            for bx, by in self.snake[1:]:
                grid_chars[by][bx] = "B"
            # Head
            if self.snake:
                hx, hy = self.snake[0]
                grid_chars[hy][hx] = "H"
            # Apple
            if self.apple:
                ax, ay = self.apple
                grid_chars[ay][ax] = "A"

            border = "+" + "-" * self.grid_size + "+"
            lines = [border]
            for row in grid_chars:
                lines.append("|" + "".join(row) + "|")
            lines.append(border)
            output = "\n".join(lines)

            if self.render_mode == "human":
                print(output)
            return output

        raise ValueError(f"Unsupported render_mode: {self.render_mode}")
