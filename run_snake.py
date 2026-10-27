import time
from collections import deque
from src.environments.snake_env import SnakeEnv, SnakeAction, ACTION_DIRECTIONS

def get_path_to_apple(env):
    if not env.snake or not env.apple:
        return []
        
    head = env.snake[0]
    apple = env.apple
    queue = deque([(head, [])])
    visited = {head}
    
    # Treat current body as obstacles
    obstacles = set(env.snake[:-1])

    while queue:
        curr, path = queue.popleft()
        if curr == apple:
            return path
        
        for action in [SnakeAction.UP, SnakeAction.DOWN, SnakeAction.LEFT, SnakeAction.RIGHT]:
            dx, dy = ACTION_DIRECTIONS[action]
            nx, ny = curr[0] + dx, curr[1] + dy
            if 0 <= nx < env.grid_size and 0 <= ny < env.grid_size:
                if (nx, ny) not in visited and (nx, ny) not in obstacles:
                    visited.add((nx, ny))
                    queue.append(((nx, ny), path + [action]))
    return []

env = SnakeEnv(render_mode='human', random_spawn=True)
env.reset()
done = False
apple_eaten = False

print("Starting game, navigating snake towards the apple...")

while not done and not apple_eaten:
    path = get_path_to_apple(env)
    if not path:
        # If no path is found, fallback to random action
        action = env.action_space.sample()
    else:
        action = path[0]
        
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    apple_eaten = info.get("apple_eaten", False)
    time.sleep(0.5)

if apple_eaten:
    print("\nThe snake successfully ate the apple!")
else:
    print("\nThe snake crashed before it could eat the apple.")
