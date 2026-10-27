import gymnasium as gym
import numpy as np

# IMPORTANT: Import ale_py to register ALE namespace
try:
    import ale_py
except ImportError:
    print("Warning: ale_py not installed. Install with: pip install ale-py")

class PacManEnv:
    """Wrapper for Atari Ms. Pac-Man environment."""
    
    def __init__(self, render_mode=None):
        """Initialize Pac-Man environment."""
        try:
            # Use Atari Ms. Pac-Man v5
            self.env = gym.make(
                'ALE/MsPacman-v5',
                render_mode=render_mode,
                full_action_space=False
            )
        except Exception as e:
            print(f"Error creating ALE/MsPacman-v5: {e}")
            print("Trying alternative registration method...")
            # Alternative: register manually
            self.env = gym.make('MsPacman-v0')
        
        self.action_space = self.env.action_space
        self.num_actions = self.action_space.n
        self.observation_space = self.env.observation_space
        
        # Action mapping
        self.action_names = {
            0: 'UP',
            1: 'DOWN', 
            2: 'LEFT',
            3: 'RIGHT'
        }
        
        self.total_reward = 0
        self.episode_steps = 0
        self.max_episode_steps = 10000
    
        
    def reset(self, seed=None):
        """Reset environment and return initial observation."""
        obs, info = self.env.reset(seed=seed)
        self.total_reward = 0
        self.episode_steps = 0
        return self._preprocess_frame(obs), info
    
    def step(self, action):
        """Execute action and return (obs, reward, done, truncated, info)."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.episode_steps += 1
        self.total_reward += reward
        
        # Terminate if max steps reached
        if self.episode_steps >= self.max_episode_steps:
            truncated = True
        
        done = terminated or truncated
        
        return (
            self._preprocess_frame(obs),
            reward,
            terminated,
            truncated,
            {**info, 'episode_return': self.total_reward}
        )
    
    def _preprocess_frame(self, frame):
        """Preprocess raw frame to (84, 84, 1)."""
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            frame = np.mean(frame, axis=2).astype(np.uint8)
        
        # Resize to 84×84
        from cv2 import resize, INTER_AREA
        frame = resize(frame, (84, 84), interpolation=INTER_AREA)
        
        # Add channel dimension
        frame = np.expand_dims(frame, axis=2)
        
        return frame.astype(np.float32) / 255.0  # Normalize
    
    def close(self):
        """Close environment."""
        self.env.close()
    
    def render(self):
        """Render current frame."""
        return self.env.render()

# Test instantiation
if __name__ == '__main__':
    env = PacManEnv(render_mode='human')
    obs, info = env.reset(seed=42)
    print(f"Initial observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {env.action_names.get(action)}, Reward: {reward}, Done: {terminated or truncated}")
        if terminated or truncated:
            break
    
    env.close()