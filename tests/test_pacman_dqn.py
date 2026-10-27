import pytest
import torch
import numpy as np
from src.agents.dqn.pacman_dqn import PacManDQN, PacManDQNAgent, ReplayBuffer

class TestPacManDQN:
    
    def test_forward_pass(self):
        """Test forward pass produces correct Q-values."""
        model = PacManDQN(input_channels=4, num_actions=4)
        state = torch.randn(1, 4, 84, 84)
        
        q_values = model(state)
        assert q_values.shape == (1, 4), f"Expected (1, 4), got {q_values.shape}"
    
    def test_get_action(self):
        """Test action selection."""
        agent = PacManDQNAgent(num_actions=4)
        state = np.random.randn(4, 84, 84).astype(np.float32)
        
        action = agent.select_action(state, greedy=True)
        assert 0 <= action < 4, f"Invalid action: {action}"
    
    def test_replay_buffer(self):
        """Test replay buffer operations."""
        buffer = ReplayBuffer(capacity=1000)
        
        # Add transitions
        for _ in range(100):
            state = np.random.randn(4, 84, 84).astype(np.float32)
            action = np.random.randint(0, 4)
            reward = np.random.randn()
            next_state = np.random.randn(4, 84, 84).astype(np.float32)
            done = np.random.randint(0, 2)
            
            buffer.add(state, action, reward, next_state, done)
        
        assert len(buffer) == 100
        
        # Sample batch
        batch = buffer.sample(32)
        states, actions, rewards, next_states, dones = batch
        assert states.shape == (32, 4, 84, 84)
        assert actions.shape == (32,)
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = PacManDQNAgent(num_actions=4)
        assert agent.epsilon == 1.0
        assert agent.total_steps == 0

if __name__ == '__main__':
    pytest.main([__file__, '-v'])