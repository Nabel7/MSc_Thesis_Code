# run_speculator_dqn.py
from pathlib import Path
import os, sys, time, json, hashlib, platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import asdict

from envs.speculator_env import SpeculatorEnv
from rl.dqn_tf import DQNConfig
from train_speeculator_dqn_tf import train
from utils.logger import CSVLogger

# ---------- per-run output dir ----------
RUN_ID  = time.strftime("%Y%m%d-%H%M%S")
OUTDIR  = Path("logs") / f"run_{RUN_ID}"
OUTDIR.mkdir(parents=True, exist_ok=True)
EP_CSV  = OUTDIR / "spec_dqn_episodes.csv"
ST_CSV  = OUTDIR / "spec_dqn_steps.csv"

# ---------- data + features ----------
BASE = Path(__file__).resolve().parent
CSV  = BASE / "data" / "speculdquantity_analysis.csv"
print("[csv]", CSV, "exists:", CSV.exists())

FEATURES = ["WindAggregatedForecast", "DemandAggregatedForecast", "-24", "-48"]
ADD_TIME = True
EXCLUDE_YEARS = (2023,)

def compute_feature_order(df, base_feats, add_time=True):
    feats = list(base_feats)
    if add_time:
        for extra in ["sin_hour","cos_hour","sin_dow","cos_dow"]:
            if extra not in feats: feats.append(extra)
    return feats

# ---------- build TRAIN/VAL/TEST time splits ----------
df = pd.read_csv(CSV)
df["DeliveryPeriod"] = pd.to_datetime(df["DeliveryPeriod"], errors="coerce")
df = df.sort_values("DeliveryPeriod").reset_index(drop=True)
if EXCLUDE_YEARS:
    df = df[~df["DeliveryPeriod"].dt.year.isin(EXCLUDE_YEARS)].reset_index(drop=True)

# time features for stats
hr  = df["DeliveryPeriod"].dt.hour.values.astype(np.float32)
dow = df["DeliveryPeriod"].dt.dayofweek.values.astype(np.float32)
df["sin_hour"] = np.sin(2*np.pi*hr/24.0)
df["cos_hour"] = np.cos(2*np.pi*hr/24.0)
df["sin_dow"]  = np.sin(2*np.pi*dow/7.0)
df["cos_dow"]  = np.cos(2*np.pi*dow/7.0)

n = len(df)
n_train = int(n*0.70)
n_val   = int(n*0.85)
df_train = df.iloc[:n_train].copy()
df_val   = df.iloc[n_train:n_val].copy()
df_test  = df.iloc[n_val:].copy()

feats_order = compute_feature_order(df, FEATURES, add_time=ADD_TIME)
mu = df_train[feats_order].mean()
sd = df_train[feats_order].std().replace(0, 1)

TRAIN_CSV = BASE / "data" / "speculator_train.csv"
VAL_CSV   = BASE / "data" / "speculator_val.csv"
TEST_CSV  = BASE / "data" / "speculator_test.csv"

# save splits (without time features to match raw CSV that env expects)
for name, frame in [("train",df_train),("val",df_val),("test",df_test)]:
    frame.drop(columns=["sin_hour","cos_hour","sin_dow","cos_dow"]).to_csv(
        {"train":TRAIN_CSV,"val":VAL_CSV,"test":TEST_CSV}[name], index=False
    )

# ---------- normalized wrapper ----------
class NormalizedEnv:
    def __init__(self, env, feats, mu, sd):
        self.env, self.feats = env, feats
        self.mu = mu.astype("float32"); self.sd = sd.astype("float32")
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        for attr in ["DAM","BM","REF","side","q","episode_len","rng","reward_scale"]:
            if hasattr(env, attr): setattr(self, attr, getattr(env, attr))
    def _norm(self, obs):
        return ((obs - self.mu.values) / self.sd.values).astype("float32")
    def reset(self): return self._norm(self.env.reset())
    def step(self, action):
        nxt, r, d, info = self.env.step(action)
        return self._norm(nxt), r, d, info

# ---------- envs (no internal normalization) ----------
common_kwargs = dict(
    feature_cols=FEATURES,
    price_col="EURPrices",
    bm_price_col="BMImbalancePrice",
    side_col="side",
    volume_col="abs_volume",
    exclude_years=EXCLUDE_YEARS,
    episode_len=96,
    reward_scale=0.01,
    normalize_features=False,
    use_delta_actions=True,
    delta_max=30.0,
    ref_price_col="-24",
    per_mwh_fee=0.5,
    fixed_fee=0.0,
    delta_reg=0.000,   # start at zero; increase later if you want to shrink |δ|
)

train_env_raw = SpeculatorEnv(str(TRAIN_CSV), add_time_features=ADD_TIME, **common_kwargs)
val_env_raw   = SpeculatorEnv(str(VAL_CSV),   add_time_features=ADD_TIME, **common_kwargs)
test_env_raw  = SpeculatorEnv(str(TEST_CSV),  add_time_features=ADD_TIME, **common_kwargs)

env_train = NormalizedEnv(train_env_raw, feats_order, mu, sd)
env_val   = NormalizedEnv(val_env_raw,   feats_order, mu, sd)
env_test  = NormalizedEnv(test_env_raw,  feats_order, mu, sd)

# ---------- sanity: choose δ that makes limit=DAM at t0 ----------
_ = env_train.reset()
dam = float(env_train.DAM[train_env_raw.t])
ref = float(env_train.REF[train_env_raw.t])
delta_clear = np.clip(dam - ref, env_train.action_space.low[0], env_train.action_space.high[0])
_, r, _, info = env_train.step([delta_clear])
print("[sanity]", info, "reward=", r)

# ---------- DQN config ----------
cfg = DQNConfig(
    num_bins= 21,
    delta_max=30.0,
    batch_size=128,
    buffer_size=100_000,
    per_alpha=0.6,
    per_beta_start=.4,
    per_beta_end=1,
    per_beta_steps=40_000,
    dueling=True,
    double=True,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay_steps=100_000,
    lr = 3e-4,
    tau=0.01,
    gamma=0.995,
    seed=0,
)

# ---------- train ----------
logger = CSVLogger(episode_csv=str(EP_CSV), step_csv=str(ST_CSV))
agent, returns = train(env_train=env_train, env_val=env_val,
                       episodes=100, steps_per_ep=48,
                       cfg=cfg, warmup_steps=1000,
                       logger=logger, run_id=RUN_ID)
print("TRAIN returns:", returns)

# ---------- plots ----------
df_ep = pd.read_csv(EP_CSV)
plt.figure()
dtr = df_ep[df_ep["split"]=="train"]; dvl = df_ep[df_ep["split"]=="val"]
plt.plot(dtr["episode"], dtr["ep_return"], label="Train")
if not dvl.empty: plt.plot(dvl["episode"], dvl["ep_return"], label="Val")
if len(dtr) > 4:  plt.plot(dtr["episode"], dtr["ep_return"].rolling(5).mean(), "--", label="Train (roll=5)")
plt.xlabel("Episode"); plt.ylabel("Episode return")
plt.title("Speculator DQN: returns"); plt.legend(); plt.grid(True, alpha=.3)
plt.tight_layout(); plt.savefig(OUTDIR / "ep_returns.png"); print("[plot]", OUTDIR/"ep_returns.png")

if {"clear_rate_buy","clear_rate_sell"}.issubset(dtr.columns):
    plt.figure()
    plt.plot(dtr["episode"], dtr["clear_rate_buy"], label="Buy clear %")
    plt.plot(dtr["episode"], dtr["clear_rate_sell"], label="Sell clear %")
    plt.xlabel("Episode"); plt.ylabel("Clear rate (%)")
    plt.title("Speculator DQN: clear rates (train)")
    plt.legend(); plt.grid(True, alpha=.3)
    plt.tight_layout(); plt.savefig(OUTDIR / "clear_rates.png"); print("[plot]", OUTDIR/"clear_rates.png")

# ---------- manifest ----------
def sha256_12(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1<<20), b""): h.update(b)
    return h.hexdigest()[:12]

def date_range_size(csv_path: Path):
    d = pd.read_csv(csv_path, usecols=["DeliveryPeriod"], parse_dates=["DeliveryPeriod"])
    return str(d["DeliveryPeriod"].min()), str(d["DeliveryPeriod"].max()), int(len(d))

tr_start, tr_end, tr_n = date_range_size(TRAIN_CSV)
va_start, va_end, va_n = date_range_size(VAL_CSV)
te_start, te_end, te_n = date_range_size(TEST_CSV)

manifest = {
    "run_id": RUN_ID,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "tensorflow": __import__("tensorflow").__version__,
    "seed": cfg.seed,
    "data": {
        "train_csv": str(TRAIN_CSV), "train_sha256_12": sha256_12(TRAIN_CSV),
        "val_csv":   str(VAL_CSV),   "val_sha256_12":   sha256_12(VAL_CSV),
        "test_csv":  str(TEST_CSV),  "test_sha256_12":  sha256_12(TEST_CSV),
        "train_range": [tr_start, tr_end], "train_rows": tr_n,
        "val_range":   [va_start, va_end], "val_rows": va_n,
        "test_range":  [te_start, te_end], "test_rows": te_n,
        "features": FEATURES, "add_time_features": ADD_TIME,
        "exclude_years": EXCLUDE_YEARS,
        "norm_stats_from": "TRAIN_ONLY",
    },
    "env": {
        "use_delta_actions": True,
        "delta_max": cfg.delta_max,
        "ref_price_col": "-24",
        "per_mwh_fee": common_kwargs["per_mwh_fee"],
        "fixed_fee":   common_kwargs["fixed_fee"],
        "delta_reg":   common_kwargs["delta_reg"],
        "reward_scale": train_env_raw.reward_scale,
        "episode_len": train_env_raw.episode_len,
    },
    "agent": asdict(cfg),
    "training": {
        "episodes": 30, "steps_per_ep": 96,
        "warmup_steps": 2000,
        "batch_size": cfg.batch_size, "buffer_size": cfg.buffer_size
    }
}
with open(OUTDIR / "manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
print(f"[manifest] {OUTDIR/'manifest.json'}")
