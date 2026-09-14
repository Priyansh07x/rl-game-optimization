"""Visual preprocessing pipeline for Snake RGB observations.

Converts raw (240, 240, 3) RGB game frames to a single-channel (64, 64) representation
using resizing, RGB-to-HSV color transformation, and background suppression.

NOTE ON IMPLEMENTATION CHOICES:
The project guide specifies "resize to 64 x 64" and "HSV/background processing"
leading to a (4, 64, 64) DQN convention, but does not dictate exact numeric HSV
thresholds. The constants defined below (e.g. BACKGROUND_VALUE_THRESHOLD = 0.1)
are deterministic implementation choices selected for this project.
"""

from typing import Optional, Tuple, Union
import numpy as np
from PIL import Image


# Implementation choice: Thresholds for separating foreground entities from background
# Background in SnakeEnv is [15, 15, 15], which gives V ~= 0.059 and S = 0.0
DEFAULT_BACKGROUND_VALUE_THRESHOLD: float = 0.10
DEFAULT_BACKGROUND_SATURATION_THRESHOLD: float = 0.05

# Target spatial resolution required by the project specification
TARGET_SPATIAL_SHAPE: Tuple[int, int] = (64, 64)


def rgb_to_hsv_numpy(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB image with values in [0, 1] to HSV color space using NumPy.

    Args:
        rgb: Float array of shape (H, W, 3) with values in range [0, 1].

    Returns:
        HSV float array of shape (H, W, 3) where:
        - H (Hue) in [0, 360)
        - S (Saturation) in [0, 1]
        - V (Value) in [0, 1]
    """
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    delta = max_c - min_c

    # Value
    v = max_c

    # Saturation
    s = np.zeros_like(v)
    non_zero_v = v > 1e-8
    s[non_zero_v] = delta[non_zero_v] / v[non_zero_v]

    # Hue
    h = np.zeros_like(v)
    non_zero_delta = delta > 1e-8

    # When red is max
    idx_r = non_zero_delta & (max_c == r)
    h[idx_r] = (60.0 * ((g[idx_r] - b[idx_r]) / delta[idx_r])) % 360.0

    # When green is max
    idx_g = non_zero_delta & (max_c == g)
    h[idx_g] = (60.0 * ((b[idx_g] - r[idx_g]) / delta[idx_g]) + 120.0) % 360.0

    # When blue is max
    idx_b = non_zero_delta & (max_c == b)
    h[idx_b] = (60.0 * ((r[idx_b] - g[idx_b]) / delta[idx_b]) + 240.0) % 360.0

    return np.stack([h, s, v], axis=-1)


class SnakePreprocessor:
    """Preprocessor for converting raw Snake RGB observations into single-channel 64x64 frames."""

    def __init__(
        self,
        target_shape: Tuple[int, int] = TARGET_SPATIAL_SHAPE,
        bg_value_threshold: float = DEFAULT_BACKGROUND_VALUE_THRESHOLD,
        bg_sat_threshold: float = DEFAULT_BACKGROUND_SATURATION_THRESHOLD,
        mode: str = "masked_value",
        dtype: np.dtype = np.float32,
    ):
        """Initialize Snake Preprocessor.

        Args:
            target_shape: Output (height, width) spatial dimensions (default: (64, 64)).
            bg_value_threshold: Minimum HSV Value to consider non-background (implementation choice).
            bg_sat_threshold: Minimum HSV Saturation to consider non-background (implementation choice).
            mode: Processing mode:
                  - "masked_value": Background is 0.0, foreground entities retain normalized Value [0.0, 1.0].
                  - "binary": Background is 0.0, foreground is 1.0.
                  - "entity_segmented": Background=0.0, Body=0.5, Head=0.8, Apple=1.0.
            dtype: Output numpy datatype (np.float32 or np.uint8).
        """
        self.target_shape = target_shape
        self.bg_value_threshold = bg_value_threshold
        self.bg_sat_threshold = bg_sat_threshold
        self.mode = mode
        self.dtype = dtype

        if self.mode not in ("masked_value", "binary", "entity_segmented"):
            raise ValueError(f"Unknown mode: {mode}. Expected 'masked_value', 'binary', or 'entity_segmented'.")

    def validate_input(self, frame: np.ndarray) -> None:
        """Validate the input frame format.

        Raises:
            TypeError: If input is not a numpy array.
            ValueError: If input does not have 3 dimensions or 3 color channels.
        """
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(frame).__name__}")

        if frame.ndim != 3:
            raise ValueError(f"Expected 3D array (H, W, C), got ndim={frame.ndim} with shape {frame.shape}")

        if frame.shape[2] != 3:
            raise ValueError(f"Expected 3 channels (RGB) in last axis, got shape {frame.shape}")

        if frame.shape[0] <= 0 or frame.shape[1] <= 0:
            raise ValueError(f"Invalid spatial dimensions: {frame.shape}")

    def resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize RGB frame to target spatial dimensions (target_shape).

        Args:
            frame: RGB array of shape (H, W, 3).

        Returns:
            Resized RGB array of shape (target_height, target_width, 3).
        """
        if frame.shape[:2] == self.target_shape:
            return frame

        # Convert uint8 if necessary for PIL resize
        if np.issubdtype(frame.dtype, np.floating):
            uint8_img = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        else:
            uint8_img = np.clip(frame, 0, 255).astype(np.uint8)

        pil_img = Image.fromarray(uint8_img, mode="RGB")
        resized_pil = pil_img.resize((self.target_shape[1], self.target_shape[0]), Image.Resampling.BILINEAR)
        return np.array(resized_pil, dtype=np.uint8)

    def process_hsv(self, resized_rgb: np.ndarray) -> np.ndarray:
        """Apply HSV transformation and background filtering.

        Args:
            resized_rgb: RGB image array of shape (64, 64, 3) in [0, 255].

        Returns:
            Single-channel processed frame of shape (64, 64).
        """
        # Normalize RGB to [0.0, 1.0] for HSV conversion
        rgb_norm = resized_rgb.astype(np.float32) / 255.0
        hsv = rgb_to_hsv_numpy(rgb_norm)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

        # Background mask: low brightness OR low saturation
        is_foreground = (v > self.bg_value_threshold) & (s > self.bg_sat_threshold)

        if self.mode == "binary":
            processed = np.where(is_foreground, 1.0, 0.0).astype(np.float32)

        elif self.mode == "entity_segmented":
            # Apple: Red hue [0, 30] U [330, 360]
            is_apple = is_foreground & ((h < 30.0) | (h >= 330.0))
            # Snake: Green hue [80, 170]
            is_snake = is_foreground & (h >= 80.0) & (h <= 170.0)
            # Head has highest value in green range (bright green)
            is_head = is_snake & (v >= 0.85)
            is_body = is_snake & (v < 0.85)

            processed = np.zeros(self.target_shape, dtype=np.float32)
            processed[is_body] = 0.5
            processed[is_head] = 0.8
            processed[is_apple] = 1.0

        else:  # "masked_value" (default)
            # Background is suppressed to 0.0; foreground retains its Value/luminance
            processed = np.where(is_foreground, v, 0.0).astype(np.float32)

        if self.dtype == np.uint8:
            return np.clip(processed * 255.0, 0, 255).astype(np.uint8)
        return processed.astype(self.dtype)

    def transform(self, frame: np.ndarray) -> np.ndarray:
        """Complete preprocessing pipeline: validate -> resize -> HSV/background filter.

        Args:
            frame: Raw RGB observation of shape (H, W, 3).

        Returns:
            Processed single-channel frame of shape (64, 64) with configured dtype.
        """
        self.validate_input(frame)
        resized = self.resize_frame(frame)
        processed = self.process_hsv(resized)
        return processed

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """Callable shorthand for transform."""
        return self.transform(frame)


def preprocess_snake_frame(
    frame: np.ndarray,
    target_shape: Tuple[int, int] = TARGET_SPATIAL_SHAPE,
    mode: str = "masked_value",
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Convenience function to preprocess a single Snake RGB observation.

    Args:
        frame: Raw RGB array of shape (H, W, 3).
        target_shape: Target spatial dimensions (default: (64, 64)).
        mode: Processing mode ("masked_value", "binary", "entity_segmented").
        dtype: Output numpy datatype (default: np.float32).

    Returns:
        Processed 2D frame of shape (64, 64).
    """
    preprocessor = SnakePreprocessor(target_shape=target_shape, mode=mode, dtype=dtype)
    return preprocessor.transform(frame)
