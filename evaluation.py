import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import time

env = gym.make("LunarLander-v3", render_mode="human")

model = PPO.load("models/ppo_lunar_lander_v3")

N = 10
scores = []

for ep in range(N):
    obs, info = env.reset(seed=ep)
    done = False
    trunc = False
    ep_r = 0.0

    while not (done or trunc):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, trunc, info = env.step(action)
        ep_r += r

    scores.append(ep_r) 
    print(f"Episode {ep+1}: Reward = {ep_r:.1f}")

env.close()
print(f"Average Reward over {N} episodes: {np.mean(scores):.1f}")