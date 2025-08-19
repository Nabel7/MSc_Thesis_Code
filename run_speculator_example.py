# run_speculator_example.py
from pathlib import Path
import numpy as np

from envs.speculator_env import SpeculatorEnv
from train_speculator_ddpg_tf import train
from rl.ddpg_tf import DDPGConfig
from train_speeculator_dqn_tf import train
from rl.dqn_tf import DQNConfig

# Resolve CSV relative to this file so paths don't break 
BASE = Path(__file__).resolve().parent
CSV = BASE / "data" / "speculator_per_side.csv"
print("[csv]", CSV, "exists:", CSV.exists())

# ---- FEATURES (state vector) ----
# Pre-gate info only: forecasts, lags, calendar (env auto-adds), and side.
FEATURES = [
    "WindAggregatedForecast",
    "DemandAggregatedForecast",
    "-24",
    "-48",
    "side",
]

# ---- Construct the environment ----
env = SpeculatorEnv(
    csv_path=str(CSV),
    feature_cols=FEATURES,
    price_col="EURPrices",            # used for reward/clearing, NOT in FEATURES
    bm_price_col="BMImbalancePrice",  # used for reward, NOT in FEATURES
    side_col="side",
    volume_col="abs_volume",          # should be 1.0 in your per-side CSV (unit MWh)
    exclude_years=(2023,),
    bid_low=0.0, bid_high=300.0,      # only used if use_delta_actions=False
    episode_len=96,
    per_mwh_fee=0.0, fixed_fee=0.0,
    reward_scale=0.01, normalize_features=True,
    use_delta_actions=True,
    delta_max=50.0,
    ref_price_col="-24",              # *** anchor delta to lag-24 ***
)

# ---- Sanity probe (delta-mode) ----
# Make limit == DAM by choosing δ = DAM - ref (guarantees clear for both sides).
o = env.reset()
dam = float(env.DAM[env.t])
ref = float(env.REF[env.t])
delta_clear = np.clip(dam - ref, env.action_space.low[0], env.action_space.high[0])
_, r, _, info = env.step([delta_clear])
print("[sanity]", info, "reward=", r)

# ---- DDPG hyperparameters ----
# Exploration noise is in €/MWh (same units as delta). Start with ~20% of delta_max.
cfg = DDPGConfig(
    batch_size=64,
    buffer_size=100_000,
    ou_sigma=10.0,           # was 0.10; too small for €/MWh deltas
    per_beta_steps=50_000,
)

# ---- Train ----
# For a quick smoke test with 2 episodes, disable warm-up so you see updates.
# For real training: increase episodes (e.g., 50–200) and set warmup_steps~5000.
agent, returns = train(env, episodes=2, steps_per_ep=96, cfg=cfg, warmup_steps=0)
print("Returns:", returns)


cfg = DQNConfig(
    num_bins=21,          # try 21 first (±50 step 5) — tune on VAL
    delta_max=50.0,
    batch_size=256,
    buffer_size=200_000,
    eps_decay_steps=200_000,
    per_beta_steps=200_000,
    dueling=True,
    double=True,
)
agent, returns = train(env, episodes=10, steps_per_ep=96, cfg=cfg, warmup_steps=2000)
print("Returns:", returns)