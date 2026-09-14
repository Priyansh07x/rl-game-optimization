"""Unit and integration tests for Snake visual preprocessing and 4-frame stacking."""

import csv
from pathlib import Path
import numpy as np
import pytest

from src.environments.snake_env import SnakeAction, SnakeEnv
from src.preprocessing.frame_stack import FrameStack
from src.preprocessing.snake_preprocess import (
    SnakePreprocessor,
    preprocess_snake_frame,
    rgb_to_hsv_numpy,
    DEFAULT_BACKGROUND_VALUE_THRESHOLD,
    DEFAULT_BACKGROUND_SATURATION_THRESHOLD,
)


class TestSnakePreprocessor:
    """Test standalone Snake RGB preprocessor logic."""

    def test_valid_rgb_input_acceptance(self):
        preprocessor = SnakePreprocessor()
        valid_frame = np.full((240, 240, 3), 15, dtype=np.uint8)
        # Put an apple and snake head
        valid_frame[20:40, 20:40] = [255, 32, 32]
        valid_frame[80:100, 80:100] = [0, 255, 64]

        out = preprocessor.transform(valid_frame)
        assert isinstance(out, np.ndarray)
        assert out.shape == (64, 64)
        assert out.dtype == np.float32

    def test_invalid_input_rejection(self):
        preprocessor = SnakePreprocessor()

        # Non-array input
        with pytest.raises(TypeError, match="Expected numpy.ndarray"):
            preprocessor.transform([[1, 2, 3]])  # type: ignore

        # 2D array (grayscale/missing channels)
        with pytest.raises(ValueError, match="Expected 3D array"):
            preprocessor.transform(np.zeros((240, 240), dtype=np.uint8))

        # 4 channels (RGBA)
        with pytest.raises(ValueError, match="Expected 3 channels"):
            preprocessor.transform(np.zeros((240, 240, 4), dtype=np.uint8))

        # 1 channel (H, W, 1)
        with pytest.raises(ValueError, match="Expected 3 channels"):
            preprocessor.transform(np.zeros((240, 240, 1), dtype=np.uint8))

        # Empty dimensions
        with pytest.raises(ValueError, match="Invalid spatial dimensions"):
            preprocessor.transform(np.zeros((0, 240, 3), dtype=np.uint8))

    def test_output_spatial_dimensions(self):
        preprocessor = SnakePreprocessor(target_shape=(64, 64))
        raw = np.random.randint(0, 256, (120, 180, 3), dtype=np.uint8)
        out = preprocessor.transform(raw)
        assert out.shape == (64, 64)

    def test_output_single_channel_representation(self):
        raw = np.zeros((240, 240, 3), dtype=np.uint8)
        out = preprocess_snake_frame(raw)
        assert out.ndim == 2
        assert out.shape == (64, 64)

    def test_determinism(self):
        preprocessor = SnakePreprocessor()
        np.random.seed(42)
        raw = np.random.randint(0, 256, (240, 240, 3), dtype=np.uint8)

        out1 = preprocessor.transform(raw)
        out2 = preprocessor.transform(raw)
        assert np.array_equal(out1, out2)

    def test_background_suppression(self):
        """Verify that dark background pixels are suppressed to 0.0."""
        preprocessor = SnakePreprocessor()
        # Snake background is [15, 15, 15]
        bg_frame = np.full((240, 240, 3), 15, dtype=np.uint8)
        out = preprocessor.transform(bg_frame)
        assert np.all(out == 0.0)

    def test_entity_segmented_mode(self):
        preprocessor = SnakePreprocessor(mode="entity_segmented")
        frame = np.full((240, 240, 3), 15, dtype=np.uint8)
        # Add apple [255, 32, 32]
        frame[0:20, 0:20] = [255, 32, 32]
        # Add snake head [0, 255, 64]
        frame[40:60, 40:60] = [0, 255, 64]
        # Add snake body [0, 180, 48]
        frame[80:100, 80:100] = [0, 180, 48]

        out = preprocessor.transform(frame)
        assert out.shape == (64, 64)
        assert np.max(out) <= 1.0
        assert np.min(out) >= 0.0

    def test_dtype_and_value_range(self):
        # float32 mode
        prep_f32 = SnakePreprocessor(dtype=np.float32)
        frame = np.random.randint(0, 256, (240, 240, 3), dtype=np.uint8)
        out_f32 = prep_f32.transform(frame)
        assert out_f32.dtype == np.float32
        assert np.all(out_f32 >= 0.0) and np.all(out_f32 <= 1.0)

        # uint8 mode
        prep_u8 = SnakePreprocessor(dtype=np.uint8)
        out_u8 = prep_u8.transform(frame)
        assert out_u8.dtype == np.uint8
        assert np.all(out_u8 >= 0) and np.all(out_u8 <= 255)


class TestFrameStack:
    """Test four-frame temporal buffer stacking."""

    def test_initialization_on_reset(self):
        stack = FrameStack(stack_size=4, frame_shape=(64, 64))
        frame0 = np.full((64, 64), 0.5, dtype=np.float32)

        stacked = stack.reset(frame0)
        assert stacked.shape == (4, 64, 64)
        assert stacked.dtype == np.float32
        # All 4 slots should contain identical initial frame copies
        for i in range(4):
            assert np.array_equal(stacked[i], frame0)

    def test_stack_size_and_dimensions(self):
        stack = FrameStack(stack_size=4, frame_shape=(64, 64))
        f0 = np.zeros((64, 64), dtype=np.float32)
        stack.reset(f0)
        assert len(stack) == 4
        assert stack.is_full
        assert stack.get_stacked().shape == (4, 64, 64)

    def test_temporal_ordering_and_sliding_window(self):
        stack = FrameStack(stack_size=4, frame_shape=(64, 64))
        f0 = np.full((64, 64), 0.0, dtype=np.float32)
        f1 = np.full((64, 64), 1.0, dtype=np.float32)
        f2 = np.full((64, 64), 2.0, dtype=np.float32)
        f3 = np.full((64, 64), 3.0, dtype=np.float32)
        f4 = np.full((64, 64), 4.0, dtype=np.float32)

        # Reset with f0 -> stack contains [f0, f0, f0, f0]
        stack.reset(f0)

        # Step f1 -> [f0, f0, f0, f1]
        s1 = stack.step(f1)
        assert s1[0, 0, 0] == 0.0
        assert s1[1, 0, 0] == 0.0
        assert s1[2, 0, 0] == 0.0
        assert s1[3, 0, 0] == 1.0

        # Step f2 -> [f0, f0, f1, f2]
        s2 = stack.step(f2)
        assert s2[0, 0, 0] == 0.0
        assert s2[1, 0, 0] == 0.0
        assert s2[2, 0, 0] == 1.0
        assert s2[3, 0, 0] == 2.0

        # Step f3 -> [f0, f1, f2, f3]
        s3 = stack.step(f3)
        assert s3[0, 0, 0] == 0.0
        assert s3[1, 0, 0] == 1.0
        assert s3[2, 0, 0] == 2.0
        assert s3[3, 0, 0] == 3.0

        # Step f4 -> [f1, f2, f3, f4] (oldest f0 is evicted)
        s4 = stack.step(f4)
        assert s4[0, 0, 0] == 1.0
        assert s4[1, 0, 0] == 2.0
        assert s4[2, 0, 0] == 3.0
        assert s4[3, 0, 0] == 4.0

    def test_reset_clears_old_history(self):
        stack = FrameStack(stack_size=4, frame_shape=(64, 64))
        f_old = np.full((64, 64), 9.0, dtype=np.float32)
        stack.reset(f_old)
        stack.step(f_old)

        # Reset with completely new episode frame
        f_new = np.full((64, 64), 1.0, dtype=np.float32)
        stacked = stack.reset(f_new)

        for i in range(4):
            assert np.array_equal(stacked[i], f_new)
            assert not np.any(stacked[i] == 9.0)

    def test_invalid_frame_shape_rejection(self):
        stack = FrameStack(stack_size=4, frame_shape=(64, 64))
        with pytest.raises(ValueError, match="Frame shape"):
            stack.reset(np.zeros((32, 32), dtype=np.float32))


class TestSnakePreprocessingIntegration:
    """Integration test coupling SnakeEnv RGB rendering with preprocessing and frame stacking."""

    def test_snake_env_preprocessing_and_stacking_pipeline(self):
        env = SnakeEnv(obs_type="rgb", random_spawn=False)
        preprocessor = SnakePreprocessor(target_shape=(64, 64), dtype=np.float32)
        stack = FrameStack(stack_size=4, frame_shape=(64, 64), dtype=np.float32)

        # 1. Reset environment
        raw_obs, info = env.reset(seed=42)
        assert raw_obs.shape == (240, 240, 3)

        # 2. Preprocess initial frame
        proc_frame = preprocessor.transform(raw_obs)
        assert proc_frame.shape == (64, 64)

        # 3. Reset frame stack
        stacked_state = stack.reset(proc_frame)
        assert stacked_state.shape == (4, 64, 64)

        # 4. Step environment and verify stacked shape at every step
        actions = [SnakeAction.RIGHT, SnakeAction.UP, SnakeAction.LEFT, SnakeAction.DOWN]
        for act in actions:
            next_raw, reward, terminated, truncated, step_info = env.step(act)
            next_proc = preprocessor.transform(next_raw)
            next_stacked = stack.step(next_proc)

            assert next_stacked.shape == (4, 64, 64)
            assert next_stacked.dtype == np.float32
            assert np.all(next_stacked >= 0.0) and np.all(next_stacked <= 1.0)

            if terminated or truncated:
                break

    def test_generate_preprocessing_smoke_csv_evidence(self):
        """Execute a deterministic simulation and log evidence to logs/snake_preprocessing_smoke_test.csv."""
        env = SnakeEnv(obs_type="rgb", random_spawn=True, max_steps=20)
        preprocessor = SnakePreprocessor(target_shape=(64, 64), dtype=np.float32)
        stack = FrameStack(stack_size=4, frame_shape=(64, 64), dtype=np.float32)

        records = []
        num_episodes = 3

        for ep in range(num_episodes):
            raw_obs, info = env.reset(seed=500 + ep)
            proc_frame = preprocessor.transform(raw_obs)
            stacked = stack.reset(proc_frame)

            step = 0
            done = False

            while not done and step < 15:
                step += 1
                action = int(env.action_space.sample())
                next_raw, reward, terminated, truncated, step_info = env.step(action)
                next_proc = preprocessor.transform(next_raw)
                stacked = stack.step(next_proc)
                done = terminated or truncated

                records.append({
                    "episode": ep + 1,
                    "step": step,
                    "action": action,
                    "raw_height": next_raw.shape[0],
                    "raw_width": next_raw.shape[1],
                    "raw_channels": next_raw.shape[2],
                    "processed_height": next_proc.shape[0],
                    "processed_width": next_proc.shape[1],
                    "stack_channels": stacked.shape[0],
                    "dtype": str(stacked.dtype),
                    "min_value": float(np.min(stacked)),
                    "max_value": float(np.max(stacked)),
                })

        output_path = Path("logs/snake_preprocessing_smoke_test.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "episode",
            "step",
            "action",
            "raw_height",
            "raw_width",
            "raw_channels",
            "processed_height",
            "processed_width",
            "stack_channels",
            "dtype",
            "min_value",
            "max_value",
        ]
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

        assert output_path.is_file()
        assert output_path.stat().st_size > 0
