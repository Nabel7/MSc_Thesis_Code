# run_generator_ddpg_tf.py
from pathlib import Path
import numpy as np
import pandas as pd

from envs.generator_env import ISEMGeneratorEnv, PlantParams, CostParams, EnvConfig
from rl.ddpg_tf import DDPGConfig
from train_generator_ddpg import train

# ---------- feature helpers ----------
def sanitize_features(csv_path: Path, wanted: list[str], price_col: str) -> list[str]:
    """Keep only columns that exist and are not the price column."""
    df = pd.read_csv(csv_path, nrows=1)
    feats = [c for c in wanted if c in df.columns and c != price_col]
    missing = [c for c in wanted if c not in df.columns]
    if missing:
        print("[warn] missing features (skipped):", missing)
    return feats

class NormalizedEnv:
    """
    Wrap an env and apply (obs - mu) / sd on the entire observation vector.
    mu, sd must each be shape (obs_dim,).
    """
    def __init__(self, env, mu, sd):
        self.env = env
        self.mu = np.asarray(mu, dtype=np.float32).reshape(-1)
        self.sd = np.asarray(sd, dtype=np.float32).reshape(-1)
        assert self.mu.shape[0] == env.obs_dim and self.sd.shape[0] == env.obs_dim, \
            f"mu/sd length must equal obs_dim ({env.obs_dim}); got {self.mu.shape[0]}, {self.sd.shape[0]}"

        # pass-throughs used by training
        self.obs_dim = env.obs_dim
        self.action_space = env.action_space
        self.cfg = env.cfg
        self.plant = env.plant
        self.costs = env.costs
        self.price = env.price
        self.X = env.X
        self.n = env.n

    def _norm(self, obs):
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        return ((obs - self.mu) / self.sd).astype(np.float32)

    def reset(self):
        return self._norm(self.env.reset())

    def step(self, action):
        nxt, r, d, info = self.env.step(action)
        return self._norm(nxt), r, d, info

    # convenience passthroughs
    @property
    def action_low(self):  return self.env.action_low
    @property
    def action_high(self): return self.env.action_high

def main():
    BASE = Path(__file__).resolve().parent
    TRAIN_CSV = BASE / "data" / "speculator_train.csv"
    VAL_CSV   = BASE / "data" / "speculator_val.csv"
    TEST_CSV  = BASE / "data" / "speculator_test.csv"

    # ---- pick your DAM price column ----
    price_col = "EURPrices"  # change if your CSV uses a different name

    # ---- features for a gas unit (only those that exist will be used) ----
    FEATS_WANTED = [
        "DemandAggregatedForecast",
        "WindAggregatedForecast",
        "-24", "-48",
        "GasPrice_EUR_per_MWhth",
        "CarbonPrice",
        "GasVarCost_EUR_per_MWh"   # if you keep constant var cost in the env, drop this to avoid mismatch
        ]
    feats_train = sanitize_features(TRAIN_CSV, FEATS_WANTED, price_col)
    feats_val   = sanitize_features(VAL_CSV,   FEATS_WANTED, price_col)
    feats_test  = sanitize_features(TEST_CSV,  FEATS_WANTED, price_col)

    # ---- plant & cost (CCGT-ish) ----
    plant = PlantParams(
        P_min=50.0, P_max=400.0,
        ramp_up=60.0, ramp_down=60.0,   # MW per 30-min step
        min_up_steps=4, min_down_steps=4,
        start_cost=10_000.0,
        no_load_cost_per_hour=600.0,
        ramp_cost_per_MW=0.0,
    )
    costs = CostParams(variable_cost_per_MWh=40.0)

    # IMPORTANT: we will compute our own normalization stats
    cfg = EnvConfig(
        dt_hours=0.5,
        episode_len=96,
        normalize_features=False,     # <- off so we can control stats
        add_time_features=True,
        time_col="DeliveryPeriod",
    )

    # 1) Base envs (no normalization inside env)
    env_train_base = ISEMGeneratorEnv(
        csv_path=str(TRAIN_CSV), price_col=price_col, feature_cols=feats_train,
        plant=plant, costs=costs, cfg=cfg,var_cost_col="GasVarCost_EUR_per_MWh",   
    )
    env_val_base = ISEMGeneratorEnv(
        csv_path=str(VAL_CSV), price_col=price_col, feature_cols=feats_val,
        plant=plant, costs=costs, cfg=cfg,var_cost_col="GasVarCost_EUR_per_MWh",   
    )
    env_test_base = ISEMGeneratorEnv(
        csv_path=str(TEST_CSV), price_col=price_col, feature_cols=feats_test,
        plant=plant, costs=costs, cfg=cfg,var_cost_col="GasVarCost_EUR_per_MWh",   
    )

    print(f"[env] obs_dim: {env_train_base.obs_dim} | feature_dim (X): {env_train_base.X.shape[1]} | price_col: {price_col}")
    print(f"[env] features used (train): {feats_train}")

    # 2) Compute normalization stats from TRAIN features (env adds 4 time features itself)
    d_feat = env_train_base.X.shape[1]               # feature columns incl. time features
    mu_feat = env_train_base.X.mean(axis=0)
    sd_feat = env_train_base.X.std(axis=0) + 1e-8

    # pad by +2 for [prev_P/Pmax, on_flag] which we center at (0,0) with unit scale
    mu_obs = np.concatenate([mu_feat, [0.0, 0.0]])
    sd_obs = np.concatenate([sd_feat, [1.0, 1.0]])

    print(f"[norm] mu_obs len: {len(mu_obs)} | sd_obs len: {len(sd_obs)}")

    # 3) Wrap base envs with NormalizedEnv
    env_train = NormalizedEnv(env_train_base, mu_obs, sd_obs)
    env_val   = NormalizedEnv(env_val_base,   mu_obs, sd_obs)
    env_test  = NormalizedEnv(env_test_base,  mu_obs, sd_obs)

    # 4) DDPG config — MUST match rl/ddpg_tf.py (no 'lr' or 'ou_mu')
    ddpg_cfg = DDPGConfig(
        actor_lr=3e-4,
        critic_lr=5e-4,
        batch_size=128,
        buffer_size=100_000,
        per_alpha=0.6,
        per_beta_start=0.4,
        per_beta_end=1.0,
        per_beta_steps=200_000,
        ou_theta=0.15,
        # exploration noise scale in **MW** (matches your action units)
        ou_sigma=80,
        gamma=0.99,
        tau=5e-3,
        seed=0,
    )

    # 5) Train (with periodic validation/test evaluation)
    agent, ep_returns = train(
        env=env_train,
        episodes=80,
        steps_per_ep=96,
        cfg=ddpg_cfg,
        warmup_steps=1_000,
        log_dir=BASE / "runs_generator_ddpg",
        env_val=env_val, env_test=env_test,
        eval_val_eps=2, eval_test_eps=1,
        eval_every=10,
        manifest_data=dict(
            train_csv=str(TRAIN_CSV), val_csv=str(VAL_CSV), test_csv=str(TEST_CSV),
            price_col=price_col, features=feats_train
        ),
    )

    print("Returns:", ep_returns)

if __name__ == "__main__":
    main()
