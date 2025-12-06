import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

def make_env(seed_offset=0):
    def _f():
        env = gym.make("LunarLander-v3")
        env.reset(seed=seed_offset)
        return Monitor(env)
    return _f

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("tb_lander", exist_ok=True)


    n_envs = 8
    env = SubprocVecEnv([make_env(i) for i in range(n_envs)])

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        verbose=1,
        tensorboard_log="tb_lander/",  
        seed=0

    )

    model.learn(total_timesteps=1_000_000)
    model.save("models/ppo_lunar_lander_v3")

    env.close()
    print("Training completed and model saved.")