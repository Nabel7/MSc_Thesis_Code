
# Diagnostics toolkit

## Files
- `utils/logger.py` — tiny CSV logger you can import in your trainers.
- `plots/diagnostics.py` — reads the CSV logs and outputs figures into an output folder (default `reports/`).

## How to wire the logger (DQN example)
```python
from utils.logger import CSVLogger, EpisodeLog
logger = CSVLogger(episode_csv="logs/spec_dqn_episodes.csv", step_csv="logs/spec_dqn_steps.csv")  # step_csv optional

# inside training loop, per step (optional but great for diagnostics)
logger.log_step(split="train", episode=ep, t=t,
                DAM=info["DAM"], BM=info["BM"], ref=info["ref"],
                side=info["side"], q=info["q"],
                delta=info["delta"], limit=info["limit"],
                cleared=info["cleared"], reward=float(rew))

# end of episode
logger.log_episode(EpisodeLog(split="train", episode=ep, steps=t+1,
                              ep_return=float(ep_ret),
                              clear_rate_buy=float(clear_rate_buy),
                              clear_rate_sell=float(clear_rate_sell),
                              loss=float(last_loss or 0.0),
                              eps=float(current_eps), beta=float(current_beta)))
```

Do the same for validation episodes (`split="val"`).

## Generate the plots
```bash
python plots/diagnostics.py --episode_csv logs/spec_dqn_episodes.csv --step_csv logs/spec_dqn_steps.csv --out reports --window 10
```
If you skip step-level logging, omit `--step_csv` and you'll still get the training/val curves.
