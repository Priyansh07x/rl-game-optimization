"""Unit tests for Flappy Bird Environment and Discretizer."""

import pytest
import numpy as np
import gymnasium as gym

from src.environments.flappy_env import FlappyEnv, FlappyAction, PROJECT_TO_NATIVE_ACTION
from src.preprocessing.flappy_preprocess import FlappyDiscretizer


def test_flappy_discretizer_total_states():
    """Verify discretizer produces exactly 75 unique discrete states."""
    discretizer = FlappyDiscretizer()

    assert discretizer.num_dx_bins == 5
    assert discretizer.num_dy_bins == 5
    assert discretizer.num_vy_bins == 3
    assert discretizer.num_states == 75

    # Check bidirectional encoding/decoding mapping across all 75 indices
    seen_states = set()
    for x_bin in range(5):
        for y_bin in range(5):
            for v_bin in range(3):
                idx = discretizer.encode_state(x_bin, y_bin, v_bin)
                assert 0 <= idx < 75
                seen_states.add(idx)

                decoded_x, decoded_y, decoded_v = discretizer.decode_state(idx)
                assert (decoded_x, decoded_y, decoded_v) == (x_bin, y_bin, v_bin)

    assert len(seen_states) == 75


def test_flappy_discretizer_feature_extraction():
    """Verify feature extraction and discretization from sample observation vector."""
    discretizer = FlappyDiscretizer()

    # Sample observation vector matching schema
    # obs[0] = pipe0.x = 0.50 -> dx = 0.50 - 0.20 = 0.30
    # obs[1] = top_y = 0.20, obs[2] = bot_y = 0.40 -> gap_center = 0.30
    # obs[9] = bird_y = 0.30 -> dy = 0.30 - 0.30 = 0.0
    # obs[10] = vel_y = -0.50 -> v_y = -0.50
    obs = np.array([
        0.50, 0.20, 0.40,  # pipe 0
        1.00, 0.00, 1.00,  # pipe 1
        1.00, 0.00, 1.00,  # pipe 2
        0.30, -0.50, 0.00   # player (y, vel_y, rot)
    ], dtype=np.float64)

    dx, dy, v_y = discretizer.extract_features(obs)
    assert np.isclose(dx, 0.30)
    assert np.isclose(dy, 0.00)
    assert np.isclose(v_y, -0.50)

    state_idx, bin_tuple, cont_feats = discretizer.process_obs(obs)
    assert 0 <= state_idx < 75
    assert not np.isnan(cont_feats).any()
    assert not np.isinf(cont_feats).any()


def test_flappy_env_creation_and_reset():
    """Verify FlappyEnv initializes, resets properly, and returns valid obs & info."""
    env = FlappyEnv(survival_reward=0.5, pipe_pass_reward=5.0, collision_reward=-1000.0)

    obs, info = env.reset(seed=42)

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (12,)
    assert obs.dtype == np.float64
    assert not np.isnan(obs).any()
    assert not np.isinf(obs).any()

    # Check info dictionary contract
    assert "state_index" in info
    assert 0 <= info["state_index"] < 75
    assert info["score"] == 0
    assert info["steps"] == 0

    env.close()


def test_flappy_env_action_mapping():
    """Verify action mapping: 0 -> native 1 (FLAP), 1 -> native 0 (NO_FLAP)."""
    assert PROJECT_TO_NATIVE_ACTION[FlappyAction.FLAP] == 1
    assert PROJECT_TO_NATIVE_ACTION[FlappyAction.NO_FLAP] == 0

    env = FlappyEnv()
    obs, info = env.reset(seed=42)

    # Step action 0 (FLAP)
    obs_flap, reward, term, trunc, info_flap = env.step(0)
    assert info_flap["project_action"] == 0
    assert info_flap["project_action_name"] == "FLAP"
    assert info_flap["native_action"] == 1

    env.close()


def test_flappy_env_reward_mapping():
    """Verify custom project reward mapping: survival=+0.5, collision=-1000.0."""
    env = FlappyEnv(survival_reward=0.5, pipe_pass_reward=5.0, collision_reward=-1000.0)
    env.reset(seed=42)

    # First non-terminal step should give survival reward +0.5
    obs, reward, term, trunc, info = env.step(1)  # NO_FLAP
    if not term:
        assert reward == 0.5

    env.close()


def test_flappy_env_collision_termination():
    """Verify that idling until crash yields collision_reward (-1000.0) and terminated=True."""
    env = FlappyEnv(survival_reward=0.5, pipe_pass_reward=5.0, collision_reward=-1000.0)
    env.reset(seed=42)

    terminated = False
    last_reward = None
    step_count = 0

    # Idle (action=1 NO_FLAP) until bird hits ground/pipe
    while not terminated and step_count < 100:
        obs, reward, terminated, truncated, info = env.step(1)
        last_reward = reward
        step_count += 1

    assert terminated is True
    assert last_reward == -1000.0
    env.close()


def test_flappy_env_deterministic_reset():
    """Verify reset with fixed seed produces identical initial observation and state."""
    env1 = FlappyEnv()
    env2 = FlappyEnv()

    obs1, info1 = env1.reset(seed=123)
    obs2, info2 = env2.reset(seed=123)

    assert np.allclose(obs1, obs2)
    assert info1["state_index"] == info2["state_index"]

    env1.close()
    env2.close()
