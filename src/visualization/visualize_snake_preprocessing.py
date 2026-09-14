"""Deterministic visual validation utility for Snake preprocessing and frame stacking.

Captures representative game moments from SnakeEnv, applies SnakePreprocessor and FrameStack,
and saves visual artifacts (raw RGB, processed 64x64 frames, 4-frame stacks, and combined panels)
to observations/snake/ for qualitative inspection.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.environments.snake_env import SnakeAction, SnakeEnv
from src.preprocessing.frame_stack import FrameStack
from src.preprocessing.snake_preprocess import SnakePreprocessor


def save_rgb_image(array: np.ndarray, filepath: Union[str, Path]) -> None:
    """Save a (H, W, 3) uint8 or float numpy array as an RGB image."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    if np.issubdtype(array.dtype, np.floating):
        uint8_img = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    else:
        uint8_img = np.clip(array, 0, 255).astype(np.uint8)
    Image.fromarray(uint8_img, mode="RGB").save(path)


def save_grayscale_image(array: np.ndarray, filepath: Union[str, Path]) -> None:
    """Save a (H, W) float [0, 1] or uint8 [0, 255] array as a grayscale image."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    if np.issubdtype(array.dtype, np.floating):
        uint8_img = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    else:
        uint8_img = np.clip(array, 0, 255).astype(np.uint8)
    Image.fromarray(uint8_img, mode="L").save(path)


def plot_frame_stack(
    stacked_frames: np.ndarray,
    filepath: Union[str, Path],
    title: str = "Snake Preprocessed Frame Stack (4, 64, 64)",
) -> None:
    """Plot and save a 4-panel figure showing the temporal progression of stacked frames."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    labels = ["t - 3 (Oldest)", "t - 2", "t - 1", "t (Current)"]

    for i in range(4):
        ax = axes[i]
        im = ax.imshow(stacked_frames[i], cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title(f"Channel {i}: {labels[i]}", fontsize=11)
        ax.axis("off")

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label="Intensity [0.0 - 1.0]")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_side_by_side_comparison(
    raw_rgb: np.ndarray,
    processed_frame: np.ndarray,
    filepath: Union[str, Path],
    state_label: str = "Initial State",
) -> None:
    """Plot raw RGB next to processed single-channel 64x64 frame."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"Snake Observation Preprocessing — {state_label}", fontsize=13, fontweight="bold")

    # Raw RGB
    ax1.imshow(raw_rgb)
    ax1.set_title(f"Raw RGB Observation\nShape: {raw_rgb.shape}, dtype: {raw_rgb.dtype}", fontsize=10)
    ax1.axis("off")

    # Processed single frame
    im = ax2.imshow(processed_frame, cmap="viridis", vmin=0.0, vmax=1.0)
    ax2.set_title(f"Processed Frame (HSV Background Filter)\nShape: {processed_frame.shape}, dtype: {processed_frame.dtype}", fontsize=10)
    ax2.axis("off")

    fig.colorbar(im, ax=ax2, shrink=0.8, label="Processed Intensity")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def run_visual_validation(
    output_dir: Union[str, Path] = "observations/snake",
    seed: int = 42,
) -> Dict[str, Path]:
    """Execute deterministic scenario simulation and save visual evidence.

    Returns:
        Dictionary mapping artifact identifiers to saved file paths.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    generated_files: Dict[str, Path] = {}

    env = SnakeEnv(
        grid_size=12,
        map_size=240,
        initial_length=3,
        random_spawn=False,
        obs_type="rgb",
        render_mode="rgb_array",
    )
    preprocessor = SnakePreprocessor(target_shape=(64, 64), dtype=np.float32)
    stack = FrameStack(stack_size=4, frame_shape=(64, 64), dtype=np.float32)

    # =========================================================================
    # Scenario 1: Initial Reset State
    # =========================================================================
    raw_initial, info = env.reset(seed=seed)
    proc_initial = preprocessor.transform(raw_initial)
    stack.reset(proc_initial)

    path_raw_init = out_path / "raw_initial.png"
    save_rgb_image(raw_initial, path_raw_init)
    generated_files["raw_initial"] = path_raw_init

    path_proc_init = out_path / "processed_initial.png"
    save_grayscale_image(proc_initial, path_proc_init)
    generated_files["processed_initial"] = path_proc_init

    plot_side_by_side_comparison(raw_initial, proc_initial, out_path / "comparison_initial.png", "Initial State")

    # =========================================================================
    # Scenario 2: Movement Transition
    # =========================================================================
    raw_move, reward, term, trunc, info = env.step(SnakeAction.RIGHT)
    proc_move = preprocessor.transform(raw_move)
    stack.step(proc_move)

    path_proc_move = out_path / "processed_after_movement.png"
    save_grayscale_image(proc_move, path_proc_move)
    generated_files["processed_after_movement"] = path_proc_move

    # =========================================================================
    # Scenario 3: Near Apple & Apple Consumption
    # Deterministically place apple at (6, 6) in front of head at (5, 6)
    # =========================================================================
    env.apple = (6, 6)
    raw_near_apple = env.render()  # Head at (5, 6), apple at (6, 6)
    proc_near_apple = preprocessor.transform(raw_near_apple)

    path_proc_near = out_path / "processed_near_apple.png"
    save_grayscale_image(proc_near_apple, path_proc_near)
    generated_files["processed_near_apple"] = path_proc_near

    # Step RIGHT onto apple at (6, 6)
    raw_eat, r_eat, term_eat, trunc_eat, info_eat = env.step(SnakeAction.RIGHT)
    proc_eat = preprocessor.transform(raw_eat)
    stack.step(proc_eat)

    path_proc_eat = out_path / "processed_apple_eaten.png"
    save_grayscale_image(proc_eat, path_proc_eat)
    generated_files["processed_apple_eaten"] = path_proc_eat

    # Step UP to show direction turn and growth
    raw_up, r_up, term_up, trunc_up, info_up = env.step(SnakeAction.UP)
    proc_up = preprocessor.transform(raw_up)
    stacked_4frames = stack.step(proc_up)

    # Save 4-frame stack visualization
    path_stack = out_path / "frame_stack.png"
    plot_frame_stack(stacked_4frames, path_stack, "Snake Temporal Frame Stack [t-3 to t]")
    generated_files["frame_stack"] = path_stack

    # =========================================================================
    # Scenario 4: Wall Collision / Terminal State
    # =========================================================================
    # Drive upward until hitting wall
    raw_coll = raw_up
    while not term:
        raw_coll, r, term, trunc, info_coll = env.step(SnakeAction.UP)

    proc_coll = preprocessor.transform(raw_coll)
    path_proc_coll = out_path / "processed_collision.png"
    save_grayscale_image(proc_coll, path_proc_coll)
    generated_files["processed_collision"] = path_proc_coll

    return generated_files


if __name__ == "__main__":
    generated = run_visual_validation()
    print("Visual validation complete. Generated files:")
    for key, path in generated.items():
        print(f"  - {key}: {path}")
