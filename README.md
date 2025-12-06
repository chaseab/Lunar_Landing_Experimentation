## Lunar Landing PPO using PyTorch
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Status](https://img.shields.io/badge/Status-Solved-brightgreen)
-------------------------------------------
- Actor critic PPO to solve LunarLander-v3 
- Running experiments with several parameters to learn about PPO

## Summary of Learnings
- Entropy coefficent encourages exploration
    - short term entropy for better long-term results
    - allows the model to learn faster
    - potentially scaling the reward overtime to maximize inital entropy but still stabilizing long-term? (maybe overkill)
- Stability diagnostics
    - Approx KL tracking how far the policy moves each update (ideally it moves more to begin with a stablizes)
        - In the provided graph, the baseline was rewarded for exploring and thus moved more at the begining. However it learned nad stabilized where the blue sat for a peroid of time before exploring and never quite tapered off as well as the baseline.
    - Clip fraction measuring how often the sample updates were clipped. Indicative of whether the model is firstly trying to explore and secondly if its jumping around too much and subsequently hurting learning.
        - Ideally around 10-30% which is roughly where both trials stabilized however the baseline trial moved to ~.15 sooner than the alternative likely due to the incentive to explore.
    - Explained variance measures how well the value head fits returns
        - By subtracting the varience of the error of the prediction over the variance of the target from 1, this value reflects how well the model is doing with its predictions. In the graph, the grey baseline achieves better predictions sooner than the alternative. 
    - Entropy annealing scales the entropy coefficient by linearly decaying its weight over the course of training. In the provided graphes, the entropy maintains higher levels in the begining and middle portions of the training while tightening up at the later portions. 

## Repo Map
ppo_from_scratch.py # training loop (rollout, GAE, PPO update, TensorBoard)
eval_lander.py # watch the agent (render_mode="human")
eval_record.py # save MP4s to results/videos/
sanity_check_gym.py # quick env sanity test
results/curves/ # exported PNGs (TensorBoard screenshots)

## How to Run

# (Windows PowerShell)
. .venv\Scripts\Activate.ps1
python sanity_check_gym.py
python ppo_from_scratch.py          # training (writes tb_scratch/, models/)
python eval_lander.py               # renders a few eval episodes
python eval_record.py               # writes MP4s into results/videos/
tensorboard --logdir tb_scratch     # view learning curves
