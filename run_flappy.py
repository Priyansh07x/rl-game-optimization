import time
import gymnasium as gym
import flappy_bird_gymnasium

# FlappyBird-v0 or FlappyBird-v0
env = gym.make("FlappyBird-v0", render_mode="human")
env.reset()

done = False
print("Starting Flappy Bird environment with random actions...")
while not done:
    # 0 = do nothing, 1 = flap
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    # Small sleep to limit frame rate roughly
    time.sleep(1 / 30.0)

env.close()
print("Episode finished!")
