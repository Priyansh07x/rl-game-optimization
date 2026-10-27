import pytest
import numpy as np
from src.environments.pacman.environment import PacManEnv

class TestPacManEnv:
    
    @pytest.fixture
    def env(self):
        """Create Pac-Man environment."""
        return PacManEnv()
    
    def test_reset(self, env):
        """Test reset returns correct shape."""
        obs, info = env.reset(seed=42)
        assert obs.shape == (84, 84, 1), f"Expected (84, 84, 1), got {obs.shape}"
        assert obs.dtype == np.float32
        assert obs.min() >= 0 and obs.max() <= 1, "Observation not normalized"
    
    def test_step(self, env):
        """Test step returns valid tuple."""
        env.reset(seed=42)
        obs, reward, terminated, truncated, info = env.step(0)
        
        assert obs.shape == (84, 84, 1)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert 'episode_return' in info
    
    def test_action_space(self, env):
        """Test action space is correct."""
        env.reset(seed=42)
        assert env.action_space.n == 9, f"Expected 9 actions, got {env.action_space.n}"
    
    def test_episode_termination(self, env):
        """Test episode terminates within max steps."""
        obs, info = env.reset(seed=42)
        steps = 0
        max_steps = env.max_episode_steps + 100
        
        while steps < max_steps:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            if terminated or truncated:
                break
        
        assert steps <= env.max_episode_steps + 1, "Episode did not terminate"
    
    def test_reward_accumulation(self, env):
        """Test reward accumulation is tracked."""
        obs, info = env.reset(seed=42)
        
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert 'episode_return' in info
            if terminated or truncated:
                break
    
    def test_seed_reproducibility(self, env):
        """Test same seed produces same initial state."""
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        
        assert np.allclose(obs1, obs2), "Same seed did not produce same observation"

if __name__ == '__main__':
    pytest.main([__file__, '-v'])