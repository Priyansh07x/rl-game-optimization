"""Four-frame history stacking utility for visual reinforcement learning observations.

Maintains a sliding temporal window of 4 processed 2D frames (e.g. 64x64), producing
a stacked observation of shape (4, 64, 64) matching DQN input requirements.
"""

from collections import deque
from typing import Optional, Tuple
import numpy as np


class FrameStack:
    """Buffer that stacks consecutive 2D frames along axis 0 to provide temporal context."""

    def __init__(
        self,
        stack_size: int = 4,
        frame_shape: Tuple[int, int] = (64, 64),
        dtype: np.dtype = np.float32,
    ):
        """Initialize the FrameStack buffer.

        Args:
            stack_size: Number of frames to stack (default: 4).
            frame_shape: Expected 2D spatial dimensions (height, width) of each frame (default: (64, 64)).
            dtype: Data type for the stacked output array (default: np.float32).
        """
        self.stack_size = stack_size
        self.frame_shape = frame_shape
        self.dtype = dtype

        self._frames: deque[np.ndarray] = deque(maxlen=stack_size)

    def _validate_frame(self, frame: np.ndarray) -> None:
        """Validate input frame shape."""
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(frame).__name__}")

        if frame.shape != self.frame_shape:
            raise ValueError(
                f"Frame shape {frame.shape} does not match configured frame_shape {self.frame_shape}"
            )

    def reset(self, initial_frame: np.ndarray) -> np.ndarray:
        """Reset the stack for a new episode and fill all slots with initial_frame.

        Args:
            initial_frame: First processed frame of shape (H, W).

        Returns:
            Stacked array of shape (stack_size, H, W).
        """
        self._validate_frame(initial_frame)
        self._frames.clear()

        # Fill all slots with the initial observation to prevent cross-episode leakage
        for _ in range(self.stack_size):
            self._frames.append(np.array(initial_frame, dtype=self.dtype, copy=True))

        return self.get_stacked()

    def step(self, frame: np.ndarray) -> np.ndarray:
        """Add a new frame to the stack, dropping the oldest.

        Args:
            frame: Latest processed frame of shape (H, W).

        Returns:
            Updated stacked array of shape (stack_size, H, W).
        """
        self._validate_frame(frame)

        if len(self._frames) == 0:
            return self.reset(frame)

        self._frames.append(np.array(frame, dtype=self.dtype, copy=True))
        return self.get_stacked()

    def append(self, frame: np.ndarray) -> np.ndarray:
        """Alias for step()."""
        return self.step(frame)

    def get_stacked(self) -> np.ndarray:
        """Return the current stacked frames as a numpy array with shape (stack_size, H, W).

        Order: Index 0 is the oldest frame (t - 3), Index 3 is the newest frame (t).
        """
        if len(self._frames) < self.stack_size:
            raise RuntimeError(
                f"FrameStack contains only {len(self._frames)} frames; expected {self.stack_size}. Call reset() first."
            )

        return np.stack(list(self._frames), axis=0).astype(self.dtype)

    def clear(self) -> None:
        """Clear all stored frames."""
        self._frames.clear()

    @property
    def is_full(self) -> bool:
        """Check whether the stack contains the full number of frames."""
        return len(self._frames) == self.stack_size

    def __len__(self) -> int:
        return len(self._frames)
