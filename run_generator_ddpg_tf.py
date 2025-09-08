# run_generator_ddpg_tf.py
import numpy as np
import pandas as pd
from pathlib import Path

from envs.generator_env import ISEMGeneratorEnv, PlantParams, CostParams, EnvConfig
from rl.ddpg_tf import DDPGConfig
from train_generator_ddpg import train


# ----------------- feature helpers -----------------
def add_time_feats(df: pd.DataFrame) -> pd.DataFrame:
    """(Unused here since the env can add time features itself; kept for reference.)"""
    df = df.copy()
    if "DeliveryPeriod" in df.columns:
        t = pd.to_datetime(df["DeliveryPeriod"], errors="coerce")
        hr = t.dt.hour.values.astype(np.float32)
        dow = t.dt.dayofweek.values.astype(np.float32)
        df["sin_hour"] = np.sin(2 * np.pi * hr / 24.0)
        df["cos_hour"] = np.cos(2 * np.pi * hr / 24.0)
        df["sin_dow"] = np.sin(2 * np.pi * dow / 7.0)
        df["cos_dow"] = np.cos(2 * np.pi * dow / 7.0)
    return df


class NormalizedEnv:
    """
    Wrap an env and apply z-score normalization to observations: (obs - mu) / sd
    mu, sd must be arrays of length == env.obs_dim (features + [prev_P/Pmax, on_flag]).
    """
    def __init__(self, env, mu, sd):
        self.env = env
        self.mu = np.asarray(mu, dtype=np.float32)
        self.sd = np.asarray(sd, dtype=np.float32)
        assert self.mu.shape == (env.obs_dim,), f"mu len {len(self.mu)} != obs_dim {env.obs_dim}"
        assert self.sd.shape == (env.obs_dim,), f"sd len {len(self.sd)} != obs_dim {env.obs_dim}"

        # passthroughs
        self.obs_dim = env.obs_dim
        self.action_space = env.action_space
        self.cfg = env.cfg
        self.plant = env.plant
        self.costs = env.costs
        self.price = env.price
        self.X = env.X
        self.n = env.n

    def _norm(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        return ((obs - self.mu) / self.sd).astype(np.float32)

    def reset(self):
        return self._norm(self.env.reset())

    def step(self, action):
        nxt, r, d, info = self.env.step(action)
        return self._norm(nxt), r, d, info

    @property
    def action_low(self):  return self.env.action_low
    @property
    def action_high(self): return self.env.action_high


def sanitize_features(csv_path: Path, wanted: list[str], price_col: str) -> list[str]:
    """Keep only wanted features that exist in the CSV and are not the price column."""
    df = pd.read_csv(csv_path, nrows=1)
    feats = [c for c in wanted if c in df.columns and c != price_col]
    missing = [c for c in wanted if c not in df.columns]
    if missing:
        print("[warn] missing features (skipped):", missing)
    return feats


def main():
    BASE = Path(__file__).resolve().parent
    TRAIN_CSV = BASE / "data" / "speculator_train.csv"
    VAL_CSV   = BASE / "data" / "speculator_val.csv"
    TEST_CSV  = BASE / "data" / "speculator_test.csv"

    # ---- choose your DAM price column name as it appears in the CSVs ----
    price_col = "EURPrices"  # change if your files use a different name (e.g., "DAM")

    # ---- Gas CCGT feature shortlist (will be filtered to those present) ----
    FEATS_WANTED = [
        "DemandAggregatedForecast",
        "WindAggregatedForecast",
        "-24", "-48",
        # add if present: "FuelNBP", "CO2", "InterconnectorFlow", ...
    ]
    feats_train = sanitize_features(TRAIN_CSV, FEATS_WANTED, price_col)
    feats_val   = sanitize_features(VAL_CSV,   FEATS_WANTED, price_col)
    feats_test  = sanitize_features(TEST_CSV,  FEATS_WANTED, price_col)

    # ---- Plant and cost parameters (CCGT-ish) ----
    plant = PlantParams(
        P_min=50.0, P_max=400.0,
        ramp_up=60.0, ramp_down=60.0,   # MW per 30-min step
        min_up_steps=4, min_down_steps=4,
        start_cost=10_000.0,
        no_load_cost_per_hour=600.0,
        ramp_cost_per_MW=0.0,
    )
    costs = CostParams(variable_cost_per_MWh=40.0)

    # Turn OFF internal normalization; we’ll provide (train-only) stats via the wrapper.
    cfg = EnvConfig(
        dt_hours=0.5,
        episode_len=96,
        normalize_features=False,
        add_time_features=True,
        time_col="DeliveryPeriod",
    )

    # ---- Base envs (time features auto-added by env when add_time_features=True) ----
    env_train_base = ISEMGeneratorEnv(
        csv_path=str(TRAIN_CSV), price_col=price_col, feature_cols=feats_train,
        plant=plant, costs=costs, cfg=cfg,
    )
    env_val_base = ISEMGeneratorEnv(
        csv_path=str(VAL_CSV), price_col=price_col, feature_cols=feats_val,
        plant=plant, costs=costs, cfg=cfg,
    )
    env_test_base = ISEMGeneratorEnv(
        csv_path=str(TEST_CSV), price_col=price_col, feature_cols=feats_test,
        plant=plant, costs=costs, cfg=cfg,
    )

    # ---- Compute normalization stats from TRAIN features (env already appended 4 time features) ----
    d_feat = env_train_base.X.shape[1]          # features (including time features if enabled)
    mu_feat = env_train_base.X.mean(axis=0)
    sd_feat = env_train_base.X.std(axis=0) + 1e-8

    # +2 for [prev_P/Pmax, on_flag] in the observation
    mu_obs = np.concatenate([mu_feat, [0.0, 0.0]])
    sd_obs = np.concatenate([sd_feat, [1.0, 1.0]])

    # Quick sanity prints
    print("[env] obs_dim:", env_train_base.obs_dim,
          "| feature_dim (X):", d_feat,
          "| price_col:", price_col)
    print("[env] features used (train):", env_train_base.feature_cols)
    print("[norm] mu_obs len:", len(mu_obs), "| sd_obs len:", len(sd_obs))

    # ---- Wrap with normalization ----
    env_train = NormalizedEnv(env_train_base, mu_obs, sd_obs)
    env_val   = NormalizedEnv(env_val_base,   mu_obs, sd_obs)
    env_test  = NormalizedEnv(env_test_base,  mu_obs, sd_obs)

    # ---- DDPG config ----
    # Use a single lr unless your rl/ddpg_tf.py supports separate actor_lr/critic_lr.
    ddpg_cfg = DDPGConfig(
        lr=1e-3,
        batch_size=64,
        buffer_size=100_000,
        per_alpha=0.6,
        per_beta_start=0.4,
        per_beta_end=1.0,
        per_beta_steps=200_000,
        ou_mu=0.0,
        ou_sigma=30.0,    # MW exploration noise; try 20–50 and tune on VAL
        ou_theta=0.15,
        gamma=0.99,
        tau=5e-3,
        grad_clip_norm=10.0,
        seed=0,
    )

    # ---- Train + periodic evals ----
    agent, ep_returns = train(
        env=env_train,
        episodes=50,
        steps_per_ep=96,
        cfg=ddpg_cfg,
        warmup_steps=5_000,
        log_dir=BASE / "runs_generator_ddpg",
        env_val=env_val,
        env_test=env_test,
        eval_val_eps=2,
        eval_test_eps=0,
        eval_every=10,
        manifest_data={
            "price_col": price_col,
            "feats_train": env_train_base.feature_cols,
            "feats_val": env_val_base.feature_cols,
            "feats_test": env_test_base.feature_cols,
            "plant": plant.__dict__,
            "costs": costs.__dict__,
            "cfg": cfg.__dict__,
        }
    )

    print("Returns:", ep_returns)


if __name__ == "__main__":
    main()
