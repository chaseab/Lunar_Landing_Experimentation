import gymnasium as gym
import numpy as np

env = gym.make("LunarLander-v3")

print("Observation space:", env.action_space)
print("Action space:", env.action_space)

num_episodes = 5
for ep in range(num_episodes):
    obs, info = env.reset(seed=ep)
    done = False
    trunc = False
    total_reward = 0
    steps = 0
    while not (done or trunc):
        action = env.action_space.sample()
        obs, reward, done, trunc, info = env.step(action)
        total_reward += reward
        steps += 1
    print(f"Episode {ep+1}: steps = {steps}, total reward = {total_reward}:.1f")

env.close()
print("Sanity check completed.")