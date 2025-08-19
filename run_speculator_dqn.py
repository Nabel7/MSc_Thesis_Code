# run_speculator_dqn.py
from pathlib import Path
import numpy as np
import pandas as pd

from envs.speculator_env import SpeculatorEnv
from rl.dqn_tf import DQNConfig
from train_speeculator_dqn_tf import train

BASE = Path(__file__).resolve().parent
CSV  = BASE / "data" / "speculator_per_side.csv"
print("[csv]", CSV, "exists:", CSV.exists())

FEATURES = ["WindAggregatedForecast", "DemandAggregatedForecast", "-24", "-48", "side"]
ADD_TIME = True
EXCLUDE_YEARS = (2023,)

# ---------- helper: compute feats order as env will -----------
def compute_feature_order(df, base_feats, add_time=True):
    feats = list(base_feats)
    if add_time:
        for extra in ["sin_hour", "cos_hour", "sin_dow", "cos_dow"]:
            if extra not in feats:
                feats.append(extra)
    return feats

# ---------- build TRAIN/VAL splits by time ----------
df = pd.read_csv(CSV)
df["DeliveryPeriod"] = pd.to_datetime(df["DeliveryPeriod"], errors="coerce")
df = df.sort_values("DeliveryPeriod").reset_index(drop=True)
if EXCLUDE_YEARS:
    years = df["DeliveryPeriod"].dt.year
    df = df[~years.isin(EXCLUDE_YEARS)].reset_index(drop=True)

# add time features (same as env) ONLY to compute normalization stats
hr  = df["DeliveryPeriod"].dt.hour.values.astype(np.float32)
dow = df["DeliveryPeriod"].dt.dayofweek.values.astype(np.float32)
df["sin_hour"] = np.sin(2*np.pi*hr/24.0)
df["cos_hour"] = np.cos(2*np.pi*hr/24.0)
df["sin_dow"]  = np.sin(2*np.pi*dow/7.0)
df["cos_dow"]  = np.cos(2*np.pi*dow/7.0)

split_idx = int(len(df) * 0.8)      # 80% train, 20% val (time-ordered)
df_train = df.iloc[:split_idx].copy()
df_val   = df.iloc[split_idx:].copy()

feats_order = compute_feature_order(df, FEATURES, add_time=ADD_TIME)

# normalization stats from TRAIN only
mu = df_train[feats_order].mean()
sd = df_train[feats_order].std().replace(0, 1)

# write split CSVs (env works from a file)
TRAIN_CSV = BASE / "data" / "speculator_train.csv"
VAL_CSV   = BASE / "data" / "speculator_val.csv"
df_train.drop(columns=["sin_hour","cos_hour","sin_dow","cos_dow"]).to_csv(TRAIN_CSV, index=False)
df_val.drop(columns=["sin_hour","cos_hour","sin_dow","cos_dow"]).to_csv(VAL_CSV, index=False)

# ---------- wrapper to apply train-normalization on-the-fly ----------
class NormalizedEnv:
    def __init__(self, env, feats, mu, sd):
        self.env = env
        self.feats = feats
        self.mu = mu.astype("float32")
        self.sd = sd.astype("float32")
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        # keep passthrough attributes used elsewhere
        for attr in ["DAM","BM","REF","side","q","episode_len","rng","reward_scale"]:
            if hasattr(env, attr):
                setattr(self, attr, getattr(env, attr))

    def _norm(self, obs):
        # obs order matches feats; apply z-score
        return ((obs - self.mu.values) / self.sd.values).astype("float32")

    def reset(self):
        o = self.env.reset()
        return self._norm(o)

    def step(self, action):
        nxt, r, d, info = self.env.step(action)
        return self._norm(nxt), r, d, info

# ---------- construct environments (env does NOT normalize internally) ----------
common_kwargs = dict(
    feature_cols=FEATURES,
    price_col="EURPrices",
    bm_price_col="BMImbalancePrice",
    side_col="side",
    volume_col="abs_volume",         # per-side CSV should be 1.0 or the per-side vol if you set it
    exclude_years=EXCLUDE_YEARS,
    episode_len=96,
    reward_scale=0.01,
    normalize_features=False,        # IMPORTANT: we normalize outside with TRAIN stats
    use_delta_actions=True,
    delta_max=50.0,
    ref_price_col="-24",
)

train_env_raw = SpeculatorEnv(csv_path=str(TRAIN_CSV), add_time_features=ADD_TIME, **common_kwargs)
val_env_raw   = SpeculatorEnv(csv_path=str(VAL_CSV),   add_time_features=ADD_TIME, **common_kwargs)

env_train = NormalizedEnv(train_env_raw, feats_order, mu, sd)
env_val   = NormalizedEnv(val_env_raw,   feats_order, mu, sd)

# ---- sanity: make limit == DAM at t0 (choose δ = DAM-ref) ----
o = env_train.reset()
dam = float(env_train.DAM[train_env_raw.t])
ref = float(env_train.REF[train_env_raw.t])
delta_clear = np.clip(dam - ref, env_train.action_space.low[0], env_train.action_space.high[0])
_, r, _, info = env_train.step([delta_clear])
print("[sanity]", info, "reward=", r)

# ---------- DQN config & train on TRAIN, monitor on VAL ----------
cfg = DQNConfig(
    num_bins=21,
    delta_max=50.0,
    batch_size=256,
    buffer_size=200_000,
    eps_decay_steps=200_000,
    per_beta_steps=200_000,
    dueling=True,
    double=True,
    seed=0,
)

agent, returns = train(env_train=env_train, env_val=env_val, episodes=30, steps_per_ep=96, cfg=cfg, warmup_steps=2000)
print("TRAIN returns:", returns)

# --- Quick plots (saved to ./logs) ---
import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("logs", exist_ok=True)
ep_csv = "logs/spec_dqn_episodes.csv"
if os.path.exists(ep_csv):
    df = pd.read_csv(ep_csv)

    # 1) Train vs Val returns
    plt.figure()
    df_tr = df[df["split"] == "train"].copy()
    df_val = df[df["split"] == "val"].copy()
    plt.plot(df_tr["episode"], df_tr["ep_return"], label="Train ep_return")
    if not df_val.empty:
        plt.plot(df_val["episode"], df_val["ep_return"], label="Val ep_return")
    # smoothed train
    if len(df_tr) > 4:
        plt.plot(df_tr["episode"], df_tr["ep_return"].rolling(5).mean(), linestyle="--", label="Train (rolling=5)")
    plt.xlabel("Episode"); plt.ylabel("Episode return")
    plt.title("Speculator DQN: returns")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig("logs/ep_returns.png")
    print("[plot] saved logs/ep_returns.png")

    # 2) Clear rates by side (train only)
    if {"clear_rate_buy","clear_rate_sell"}.issubset(df_tr.columns):
        plt.figure()
        plt.plot(df_tr["episode"], df_tr["clear_rate_buy"], label="Buy clear %")
        plt.plot(df_tr["episode"], df_tr["clear_rate_sell"], label="Sell clear %")
        plt.xlabel("Episode"); plt.ylabel("Clear rate (%)")
        plt.title("Speculator DQN: clear rates (train)")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig("logs/clear_rates.png")
        print("[plot] saved logs/clear_rates.png")
else:
    print("[plot] logs/spec_dqn_episodes.csv not found — did logging run?")
