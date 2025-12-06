# eval_record.py
import os
import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

# Import ONLY the network class from your training script
from ppo_from_scratch import ActorCritic

# Local constants (don’t import these from elsewhere)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENV_ID = "LunarLander-v3"
HIDDEN = 128

def latest_checkpoint(path="models"):
    os.makedirs(path, exist_ok=True)
    ckpts = [f for f in os.listdir(path) if f.endswith(".pt")]
    if not ckpts:
        raise FileNotFoundError("No checkpoints found in ./models — let training run until one is saved.")
    ckpts.sort(key=lambda f: os.path.getmtime(os.path.join(path, f)))
    return os.path.join(path, ckpts[-1])

if __name__ == "__main__":
    # Pick checkpoint
    ckpt_path = latest_checkpoint("models")
    print("Using checkpoint:", ckpt_path)

    # Prepare video folder
    os.makedirs("results/videos", exist_ok=True)

    # Make a recordable env (rgb_array is required)
    env = gym.make(ENV_ID, render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder="results/videos",
        name_prefix="ppo_scratch_eval",
        episode_trigger=lambda ep: True,  # record every episode we run
    )

    # Build & load the model
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    net = ActorCritic(obs_dim, act_dim, hidden=HIDDEN).to(DEVICE)
    net.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    net.eval()

    # Run a couple episodes deterministically and record
    scores = []
    for ep in range(2):
        obs, info = env.reset(seed=1234 + ep)
        done = False
        trunc = False
        total = 0.0
        while not (done or trunc):
            with torch.no_grad():
                x = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                logits, _ = net(x)
                action = torch.argmax(logits, dim=-1).item()  # deterministic action
            obs, r, done, trunc, info = env.step(action)
            total += r
        scores.append(total)
        print(f"Episode {ep+1}: {total:.1f}")

    env.close()
    print("Saved videos to: results/videos")
    print("Average reward:", float(np.mean(scores)))
