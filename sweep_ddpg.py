#!/usr/bin/env python3
from __future__ import annotations
import itertools, json, csv, time
from pathlib import Path

import numpy as np

from envs.generator_env import ISEMGeneratorEnv, PlantParams, CostParams, EnvConfig
from rl.ddpg_tf import DDPGConfig
from train_generator_ddpg import train

# ---------- knobs ----------
MODE = "joint"   # "env" | "rl" | "joint"
MAX_RUNS = 60    # safety cap
EVAL_EVERY = 10
EVAL_VAL_EPS = 2
EVAL_TEST_EPS = 0

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
TRAIN_CSV = DATA_DIR / "speculator_train.csv"
VAL_CSV   = DATA_DIR / "speculator_val.csv"
TEST_CSV  = DATA_DIR / "speculator_test.csv"
LOG_ROOT  = BASE / "runs_generator_ddpg"

PRICE_COL = "EURPrices"
FEATURES  = [
    "DemandAggregatedForecast","WindAggregatedForecast","-24","-48",
    "GasPrice_EUR_per_MWhth","CarbonPrice","GasVarCost_EUR_per_MWh",
]

# If your CSV has GasVarCost_EUR_per_MWh (fuel already computed), avoid extra fixed var cost.
USE_ONLY_CSV_VARCOST = True   # set False to add a fixed var-cost €/MWh on top

# ---------- search spaces ----------

# Environment economics (keep coarse at first)
ENV_GRID = {
    # Plant
    "P_min":            [50.0, 80.0],        # (12.5%–20% of 400MW)
    "P_max":            [400.0],             # fixed
    "ramp_up":          [60.0],              # MW per 30-min step (keep fixed unless needed)
    "ramp_down":        [60.0],

    # Constraints
    "min_up_steps":     [4, 6],              # 2–8 typical; 4–6 often reasonable
    "min_down_steps":   [4, 6],

    # Costs
    "start_cost":           [5_000.0, 10_000.0, 20_000.0, 40_000.0],
    "no_load_cost_per_hour":[300.0, 600.0, 900.0],  # €/h when on
    "ramp_cost_per_MW":     [0.0, 0.5, 1.0],        # €/MW change step-to-step

    # Variable cost fallback (only used if USE_ONLY_CSV_VARCOST=False)
    "fixed_var_cost_per_MWh": [0.0] if USE_ONLY_CSV_VARCOST else [20.0, 40.0],
}

# RL / algorithmic
RL_GRID = {
    "ou_sigma":   [20.0, 40.0, 80.0],   # MW; ~{0.05,0.10,0.20} * P_max
    "tau":        [1e-3, 5e-3, 1e-2],
    "actor_lr":   [1e-4, 3e-4],
    "critic_lr":  [5e-4, 1e-3],
    "seed":       [0, 1],               # bump to 0..4 once narrowed
}

# Training meta
EPISODES = 60
WARMUP_STEPS = 1500
STEPS_PER_EP = 96

# ---------- helpers ----------
def sanitize_features(csv_path: Path, wanted: list[str], price_col: str) -> list[str]:
    import pandas as pd
    df = pd.read_csv(csv_path, nrows=1)
    feats = [c for c in wanted if c in df.columns and c != price_col]
    return feats

def make_envs(env_kwargs):
    cfg = EnvConfig(
        dt_hours=0.5,
        episode_len=STEPS_PER_EP,
        normalize_features=False,
        add_time_features=True,
        time_col="DeliveryPeriod",
    )

    feats_train = sanitize_features(TRAIN_CSV, FEATURES, PRICE_COL)
    feats_val   = sanitize_features(VAL_CSV, FEATURES, PRICE_COL)
    feats_test  = sanitize_features(TEST_CSV, FEATURES, PRICE_COL)

    plant = PlantParams(
        P_min=env_kwargs["P_min"], P_max=env_kwargs["P_max"],
        ramp_up=env_kwargs["ramp_up"], ramp_down=env_kwargs["ramp_down"],
        min_up_steps=env_kwargs["min_up_steps"], min_down_steps=env_kwargs["min_down_steps"],
        start_cost=env_kwargs["start_cost"],
        no_load_cost_per_hour=env_kwargs["no_load_cost_per_hour"],
        ramp_cost_per_MW=env_kwargs["ramp_cost_per_MW"],
    )

    # If your CSV has GasVarCost_EUR_per_MWh, likely do not add a fixed var-cost again.
    fixed_var = 0.0 if USE_ONLY_CSV_VARCOST else env_kwargs["fixed_var_cost_per_MWh"]
    costs = CostParams(variable_cost_per_MWh=fixed_var)

    env_train = ISEMGeneratorEnv(
        csv_path=str(TRAIN_CSV), price_col=PRICE_COL, feature_cols=feats_train,
        plant=plant, costs=costs, cfg=cfg, var_cost_col="GasVarCost_EUR_per_MWh",
    )
    env_val = ISEMGeneratorEnv(
        csv_path=str(VAL_CSV), price_col=PRICE_COL, feature_cols=feats_val,
        plant=plant, costs=costs, cfg=cfg, var_cost_col="GasVarCost_EUR_per_MWh",
    )
    env_test = ISEMGeneratorEnv(
        csv_path=str(TEST_CSV), price_col=PRICE_COL, feature_cols=feats_test,
        plant=plant, costs=costs, cfg=cfg, var_cost_col="GasVarCost_EUR_per_MWh",
    )
    return env_train, env_val, env_test

def make_ddpg_cfg(rl_kwargs):
    return DDPGConfig(
        actor_lr=rl_kwargs["actor_lr"],
        critic_lr=rl_kwargs["critic_lr"],
        batch_size=64,
        buffer_size=100_000,
        per_alpha=0.6,
        per_beta_start=0.4,
        per_beta_end=1.0,
        per_beta_steps=200_000,
        ou_sigma=rl_kwargs["ou_sigma"],
        ou_theta=0.15,
        gamma=0.99,
        tau=rl_kwargs["tau"],
        seed=rl_kwargs["seed"],
    )

def tag_from(env_kwargs, rl_kwargs):
    bits = [
        f"Pmin={env_kwargs['P_min']}",
        f"start={env_kwargs['start_cost']}",
        f"nlh={env_kwargs['no_load_cost_per_hour']}",
        f"rampC={env_kwargs['ramp_cost_per_MW']}",
        f"up={env_kwargs['min_up_steps']}",
        f"down={env_kwargs['min_down_steps']}",
        f"ou={rl_kwargs['ou_sigma']}",
        f"tau={rl_kwargs['tau']}",
        f"aLR={rl_kwargs['actor_lr']}",
        f"cLR={rl_kwargs['critic_lr']}",
        f"seed={rl_kwargs['seed']}",
    ]
    return "__".join(bits)

def run_one(env_kwargs, rl_kwargs, base_logdir: Path):
    env_train, env_val, env_test = make_envs(env_kwargs)
    cfg = make_ddpg_cfg(rl_kwargs)

    # simple normalization wrapper (train stats only)
    import numpy as np
    class NormalizedEnv:
        def __init__(self, env, mu, sd):
            self.env = env; self.mu=np.asarray(mu, np.float32); self.sd=np.asarray(sd, np.float32)
            self.obs_dim=env.obs_dim; self.action_space=env.action_space
            self.cfg=env.cfg; self.plant=env.plant; self.costs=env.costs; self.price=env.price; self.X=env.X; self.n=env.n
        def _n(self, x): x=np.asarray(x, np.float32); return ((x-self.mu)/self.sd).astype(np.float32)
        def reset(self): return self._n(self.env.reset())
        def step(self, a): s,r,d,i=self.env.step(a); return self._n(s),r,d,i
        @property
        def action_low(self):  return self.env.action_low
        @property
        def action_high(self): return self.env.action_high

    mu_feat = env_train.X.mean(axis=0); sd_feat = env_train.X.std(axis=0) + 1e-8
    mu_obs = np.concatenate([mu_feat, [0.0,0.0]]); sd_obs = np.concatenate([sd_feat, [1.0,1.0]])
    n_train = NormalizedEnv(env_train, mu_obs, sd_obs)
    n_val   = NormalizedEnv(env_val,   mu_obs, sd_obs)
    n_test  = NormalizedEnv(env_test,  mu_obs, sd_obs)

    tag = tag_from(env_kwargs, rl_kwargs)
    run_dir = base_logdir / f"sweep__{tag}"

    agent, ep_returns = train(
        env=n_train,
        episodes=EPISODES,
        steps_per_ep=STEPS_PER_EP,
        cfg=cfg,
        warmup_steps=WARMUP_STEPS,
        log_dir=run_dir,
        env_val=n_val,
        env_test=n_test if EVAL_TEST_EPS>0 else None,
        eval_val_eps=EVAL_VAL_EPS,
        eval_test_eps=EVAL_TEST_EPS,
        eval_every=EVAL_EVERY,
        manifest_data={
            "price_col": PRICE_COL,
            "feats_train": env_train.feature_cols,
            "feats_val": env_val.feature_cols,
            "feats_test": env_test.feature_cols,
            "plant": env_train.plant.__dict__,
            "costs": env_train.costs.__dict__,
            "cfg": env_train.cfg.__dict__,
            "mode": MODE,
        },
    )
    return run_dir

def main():
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    out_csv = LOG_ROOT / f"sweep_summary_{int(time.time())}.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tag","run_dir","last_train_return","best_val_return","mean_val_cap_factor","mean_val_starts"])

        # build spaces
        env_space = list(dict(zip(ENV_GRID, vals)) for vals in itertools.product(*ENV_GRID.values()))
        rl_space  = list(dict(zip(RL_GRID,  vals)) for vals in itertools.product(*RL_GRID.values()))

        if MODE == "env":
            combos = [(e, rl_space[0]) for e in env_space]
        elif MODE == "rl":
            combos = [(env_space[0], r) for r in rl_space]
        else:  # joint (careful with size!)
            combos = list(itertools.product(env_space, rl_space))

        # safety cap
        combos = combos[:MAX_RUNS]

        for i,(e,r) in enumerate(combos, 1):
            print(f"\n=== [{i}/{len(combos)}] {tag_from(e,r)} ===")
            run_dir = run_one(e, r, LOG_ROOT)

            # read back episodes + eval to summarize
            eps_csv = run_dir / "episodes.csv"
            eval_csv = run_dir / "eval.csv"
            last_train = ""
            best_val = ""
            mean_val_cf = ""
            mean_val_starts = ""

            try:
                import pandas as pd
                if eps_csv.exists():
                    epi = pd.read_csv(eps_csv)
                    if "episode_return" in epi.columns:
                        last_train = float(epi["episode_return"].iloc[-1])
                if eval_csv.exists():
                    ev = pd.read_csv(eval_csv)
                    if not ev.empty and "split" in ev and "mean_return" in ev:
                        val = ev[ev["split"]=="val"]
                        if not val.empty:
                            best_val = float(val["mean_return"].max())
                # pull mean val CF/starts from steps if present (approx at last eval window)
                steps_csv = run_dir / "steps.csv"
                if steps_csv.exists():
                    st = pd.read_csv(steps_csv)
                    if {"episode","on","started","P"}.issubset(st.columns):
                        last_ep = int(st["episode"].max())
                        g = st[st["episode"]==last_ep]
                        # fallback: average across all episodes if needed
                        if len(g)==0: g = st
                        Pmax = float(np.nanmax(st["P"].values)) if "P" in st.columns else np.nan
                        if Pmax>0:
                            mean_val_cf = float((g["P"].mean() / Pmax))
                        mean_val_starts = float(g["started"].sum()/max(1, g["episode"].nunique()))
            except Exception as ex:
                print("summary readback warning:", ex)

            w.writerow([tag_from(e,r), str(run_dir), last_train, best_val, mean_val_cf, mean_val_starts])

if __name__ == "__main__":
    main()
