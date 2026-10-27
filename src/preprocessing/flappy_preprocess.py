"""Flappy Bird Observation Preprocessing and State Discretization.

Extracts continuous features (dx, dy, v_y) from raw 12-element FlappyBird-v0 observations
and maps them into a 75-state discrete state space (5 x 5 x 3).
"""

from typing import Dict, List, Tuple, Union, Optional
import numpy as np


class FlappyDiscretizer:
    """Discretizes 12-element FlappyBird observations into a 75-state discrete index.

    Schema for 12-element observation (use_lidar=False):
      obs[0]: pipe0.x
      obs[1]: pipe0.top_y (bottom edge of top pipe)
      obs[2]: pipe0.bot_y (top edge of bottom pipe)
      obs[3:6]: pipe1 (x, top_y, bot_y)
      obs[6:9]: pipe2 (x, top_y, bot_y)
      obs[9]: player.y
      obs[10]: player.vel_y
      obs[11]: player.rot

    Extracted Continuous Features:
      - bird_x = 0.20 (fixed in FlappyBird-v0)
      - dx = obs[0] - bird_x (horizontal distance to upcoming pipe)
      - gap_center = (obs[1] + obs[2]) / 2.0
      - dy = gap_center - obs[9] (vertical offset: >0 means bird below gap center?
             Wait: in Pygame Y grows downwards. obs[9] is bird Y.
             If bird Y is greater than gap center Y, bird is LOWER on screen (below gap),
             so gap_center - bird_y is negative.
             Let's maintain: dy = gap_center - player_y.
             Higher dy = bird is ABOVE gap center (smaller Y).
             Lower dy = bird is BELOW gap center (larger Y).
      - v_y = obs[10] (vertical velocity: negative = moving UP, positive = falling DOWN)

    Discretization Grid:
      - dx: 5 bins
      - dy: 5 bins
      - v_y: 3 bins
      Total states: 5 * 5 * 3 = 75 states.
    """

    def __init__(
        self,
        dx_edges: Optional[List[float]] = None,
        dy_edges: Optional[List[float]] = None,
        vy_edges: Optional[List[float]] = None,
    ):
        """Initialize discretization bin boundaries.

        Args:
            dx_edges: Cutoffs for 5 dx bins (4 float values).
            dy_edges: Cutoffs for 5 dy bins (4 float values).
            vy_edges: Cutoffs for 3 v_y bins (2 float values).
        """
        # Default bin cutoffs if not provided
        # dx ranges roughly from -0.2 to 0.8
        self.dx_edges = np.array(
            dx_edges if dx_edges is not None else [0.0, 0.15, 0.30, 0.50],
            dtype=np.float64,
        )
        # dy ranges roughly from -0.5 to 0.5
        self.dy_edges = np.array(
            dy_edges if dy_edges is not None else [-0.10, -0.02, 0.02, 0.10],
            dtype=np.float64,
        )
        # v_y ranges from -1.0 (max up) to +1.0 (max down)
        self.vy_edges = np.array(
            vy_edges if vy_edges is not None else [-0.20, 0.20],
            dtype=np.float64,
        )

        self.num_dx_bins = len(self.dx_edges) + 1  # 5
        self.num_dy_bins = len(self.dy_edges) + 1  # 5
        self.num_vy_bins = len(self.vy_edges) + 1  # 3

        self.num_states = self.num_dx_bins * self.num_dy_bins * self.num_vy_bins  # 75

    def extract_features(self, obs: np.ndarray) -> Tuple[float, float, float]:
        """Extract continuous (dx, dy, v_y) from raw 12-element observation vector.

        Args:
            obs: 12-element numpy array.

        Returns:
            Tuple of (dx, dy, v_y).
        """
        if len(obs) < 12:
            raise ValueError(f"Expected observation vector of length >= 12, got {len(obs)}")

        bird_x = 0.20
        dx = float(obs[0] - bird_x)
        gap_center = float((obs[1] + obs[2]) / 2.0)
        dy = float(gap_center - obs[9])
        v_y = float(obs[10])

        return dx, dy, v_y

    def discretize_features(self, dx: float, dy: float, v_y: float) -> Tuple[int, int, int]:
        """Map continuous features (dx, dy, v_y) into bin indices.

        Args:
            dx: Horizontal distance to pipe.
            dy: Vertical distance to gap center.
            v_y: Vertical velocity.

        Returns:
            Tuple of (x_bin, y_bin, velocity_bin).
        """
        x_bin = int(np.digitize(dx, self.dx_edges))
        y_bin = int(np.digitize(dy, self.dy_edges))
        v_bin = int(np.digitize(v_y, self.vy_edges))

        # Ensure indices stay within bounds
        x_bin = min(max(x_bin, 0), self.num_dx_bins - 1)
        y_bin = min(max(y_bin, 0), self.num_dy_bins - 1)
        v_bin = min(max(v_bin, 0), self.num_vy_bins - 1)

        return x_bin, y_bin, v_bin

    def encode_state(self, x_bin: int, y_bin: int, v_bin: int) -> int:
        """Encode 3D bin tuple (x_bin, y_bin, v_bin) into a single integer state index in [0, 74].

        Formula:
          state_index = x_bin * (5 * 3) + y_bin * 3 + v_bin

        Args:
            x_bin: int in [0, 4]
            y_bin: int in [0, 4]
            v_bin: int in [0, 2]

        Returns:
            Integer state index in [0, 74].
        """
        state_idx = x_bin * (self.num_dy_bins * self.num_vy_bins) + y_bin * self.num_vy_bins + v_bin
        return int(state_idx)

    def decode_state(self, state_idx: int) -> Tuple[int, int, int]:
        """Decode a single integer state index in [0, 74] back to (x_bin, y_bin, v_bin).

        Args:
            state_idx: Integer state index in [0, 74].

        Returns:
            Tuple of (x_bin, y_bin, v_bin).
        """
        if not (0 <= state_idx < self.num_states):
            raise ValueError(f"State index {state_idx} out of range [0, {self.num_states-1}]")

        stride_x = self.num_dy_bins * self.num_vy_bins
        x_bin = state_idx // stride_x
        rem = state_idx % stride_x
        y_bin = rem // self.num_vy_bins
        v_bin = rem % self.num_vy_bins

        return x_bin, y_bin, v_bin

    def process_obs(self, obs: np.ndarray) -> Tuple[int, Tuple[int, int, int], Tuple[float, float, float]]:
        """Process raw observation vector directly into state index, bin tuple, and continuous features.

        Args:
            obs: 12-element numpy array.

        Returns:
            Tuple of (state_index, (x_bin, y_bin, v_bin), (dx, dy, v_y)).
        """
        dx, dy, v_y = self.extract_features(obs)
        x_bin, y_bin, v_bin = self.discretize_features(dx, dy, v_y)
        state_idx = self.encode_state(x_bin, y_bin, v_bin)
        return state_idx, (x_bin, y_bin, v_bin), (dx, dy, v_y)
