import os, time, math, random
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

EVN_ID = "LunarLander-v3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0

GAMMA = 0.99
LAMBDA = .95
LR = 3e-4
CLIP = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
ROLLOUT_STEPS = 2048
EPOCHS = 10
MINIBATCH = 64
MAX_STEPS = 1_000_000
HIDDEN = 128

def set_seed (seed=SEED):
    random.seed(seed); np. random.seed(seed); torch.manual_seed(seed)
    if torch.cuda. is_available(): torch.cuda.manual_seed_all(seed)

def compute_gae(rew, val, done, last_val, gamma=GAMMA, lam = LAMBDA):
    T = len(rew)
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    vals = np.append(val, last_val)
    for t in reversed(range(T)):
        delta = rew[t] + gamma * vals[t+1] * (1 - done[t]) - vals[t]
        last_gae = delta + gamma * lam * (1 - done[t]) * last_gae
        adv[t]= last_gae
    ret = adv + val
    return adv, ret


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=HIDDEN):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.pi = nn.Linear(hidden, act_dim)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.pi(h), self.v(h)



def train():
    set_seed(SEED)
    os.makedirs("models", exist_ok=True)
    writer = SummaryWriter("tb_scratch")

    env = gym.make(EVN_ID)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    net = ActorCritic(obs_dim, act_dim).to(DEVICE)
    opt = optim.Adam(net.parameters(), lr=LR)

    global_steps, update_idx = 0, 0
    obs, info = env.reset(seed=SEED)
    ep_return = 0.0
    recent_returns = []

    t0 = time.time()
    while global_steps < MAX_STEPS:
        obs_buf, act_buf, rew_buf, done_buf, logp_buf, val_buf = [], [], [], [], [], []
        steps = 0
        while steps < ROLLOUT_STEPS:
            obs_t = torch. tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            logits, v = net(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            logp = dist.log_prob(a)

            next_obs, r, done, trunc, info = env.step(a.item())
            d = float(done or trunc)

            obs_buf.append(obs)
            act_buf.append(a.item())
            rew_buf.append(r)
            done_buf.append(d)
            logp_buf.append(logp.item())
            val_buf.append(v.item())

            obs = next_obs
            steps += 1
            global_steps += 1
            ep_return += r

            if done or trunc:
                recent_returns.append(ep_return)
                writer.add_scalar("rollout/ep_return", ep_return, global_steps)
                ep_return = 0.0
                obs, info = env.reset()

        with torch.no_grad():
            obs_t_last = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            _, last_v = net(obs_t_last)
            last_v = float(last_v.item())

        rew = np.array(rew_buf, dtype=np.float32)
        val = np.array(val_buf, dtype=np.float32)
        done = np.array(done_buf, dtype=np.float32)

        adv, ret = compute_gae(rew, val, done, last_v)

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_t       = torch.tensor(np.array(obs_buf),   dtype=torch.float32, device=DEVICE)
        act_t       = torch.tensor(np.array(act_buf),   dtype=torch.long,    device=DEVICE)
        logp_old_t  = torch.tensor(np.array(logp_buf),  dtype=torch.float32, device=DEVICE)
        ret_t       = torch.tensor(np.array(ret),       dtype=torch.float32, device=DEVICE)
        adv_t       = torch.tensor(np.array(adv),       dtype=torch.float32, device=DEVICE)


        n = obs_t.size(0)
        idxs = np.arange(n)
        pi_losses, v_losses, entropies = [], [], []
        kls, clip_fracs = [], []

        for _ in range(EPOCHS):
            np.random.shuffle   (idxs)
            for start in range(0, n, MINIBATCH):
                mb = idxs[start:start+MINIBATCH]
                logits, value = net(obs_t[mb])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(act_t[mb])

                ratio = torch.exp(logp - logp_old_t[mb])
                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1.0 - CLIP, 1.0 + CLIP) * adv_t[mb]
                pi_loss = -torch.min(surr1, surr2).mean()

                v_loss = ((ret_t[mb] - value.squeeze())**2).mean()
                ent = dist.entropy().mean()

                loss = pi_loss + VF_COEF * v_loss - ENT_COEF * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()

                pi_losses.append(pi_loss.item())
                v_losses.append(v_loss.item())
                entropies.append(ent.item())
                # approx KL (old vs new) — use stop-gradient for old
                with torch.no_grad():
                    approx_kl = (logp_old_t[mb] - logp).mean().item()
                kls.append(approx_kl)

                clip_frac = (torch.abs(ratio - 1.0) > CLIP).float().mean().item()
                clip_fracs.append(clip_frac)

        update_idx += 1
        with torch.no_grad():
            v_pred = net(obs_t)[1].squeeze().cpu().numpy()
            y = ret_t.cpu().numpy()
            var_y = np.var(y)
            ev = 1.0 - np.var(y - v_pred) / (var_y + 1e-8) if var_y > 1e-12 else 0.0

        writer.add_scalar("train/explained_variance", ev, global_steps)
        writer.add_scalar("train/pi_loss", np.mean(pi_losses), global_steps)
        writer.add_scalar("train/v_loss", np.mean(v_losses), global_steps)
        writer.add_scalar("train/entropy", np.mean(entropies), global_steps)
        writer.add_scalar("train/approx_kl", np.mean(kls), global_steps)
        writer.add_scalar("train/clip_frac",  np.mean(clip_fracs), global_steps)

        if len(recent_returns) > 0:
            writer.add_scalar("eval/avg_return_5", np.mean(recent_returns[-5:]), global_steps)

        if update_idx % 10 == 0:
            torch.save(net.state_dict(), f"models/ppo_scratch_{global_steps}.pt")

        if update_idx % 5 == 0:
            dt = time.time() - t0
            avg10 = np.mean(recent_returns[-10:]) if len(recent_returns) >= 10 else float('nan')
            print(f"upd={update_idx:04d} steps={global_steps:>8} avgR(10)={avg10:6.1f} dt={dt:5.1f}s")
            t0 = time.time()


    env.close()
    writer.close()

if __name__ == "__main__":
    train()