# run_generator_ddpg_tf.py
import argparse
from pathlib import Path
import pandas as pd

from envs.generator_env import ISEMGeneratorEnv, PlantParams, CostParams, EnvConfig
from rl.ddpg_tf import DDPGConfig
from train_generator_ddpg import train

def main():
    BASE = Path(__file__).resolve().parent
    default_csv = BASE / "data" / "speculdquantity_analysis.csv"

    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=str(default_csv), help="Path to CSV (defaults to data/speculdquantity_analysis.csv)")
    p.add_argument("--price_col", default="EURPrices")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--steps_per_ep", type=int, default=96)
    p.add_argument("--log_dir", default="logs_gen")
    args = p.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)

    # Preferred, pre-gate features (only keep what exists)
    preferred = [
        "WindAggregatedForecast",
        "DemandAggregatedForecast",
        "-24", "-48",
        "BM-24", "BM-48",
        "GasVarCost_EUR_per_MWh",
        "GasPrice_EUR_per_MWhth",
        "WindShare",
        "GasCapacity_MW", "WindCapacity_MW",
        "CarbonPrice_-24",
        "InterconnectorNetTotal_-24",
    ]
    # Filter to present columns and numeric only
    feats = [c for c in preferred if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

    if not feats:
        # fallback: use all numeric except price and time
        exclude = {args.price_col, "DeliveryPeriod"}
        feats = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
        print("[warn] preferred features not found; falling back to generic numeric set.")

    # Gas unit parameters (CCGT-ish)
    plant = PlantParams(
        P_min=100.0, P_max=400.0,
        ramp_up=200.0, ramp_down=200.0,
        min_up_steps=2.0, min_down_steps=2.0,
        start_cost=20000.0, no_load_cost_per_hour=800.0,
        ramp_cost_per_MW=1.0,
    )

    # Costs — if dataset has a column for marginal cost, you can set it explicitly
    costs = CostParams(variable_cost_per_MWh=float(df.get("GasVarCost_EUR_per_MWh", pd.Series([45.0])).iloc[0]))

    cfg = EnvConfig(
        dt_hours=0.25,
        episode_len=args.steps_per_ep,   # should be 96 for 1 day
        normalize_features=True,
        add_time_features=True,
        time_col="DeliveryPeriod" if "DeliveryPeriod" in df.columns else None,
    )

    env = ISEMGeneratorEnv(
        csv_path=str(csv_path),
        price_col=args.price_col,
        feature_cols=feats,
        plant=plant,
        costs=costs,
        cfg=cfg,
    )

    # DDPG hyperparams (conservative, stable)
    ddpg = DDPGConfig(
        batch_size=128,
        buffer_size=100_000,
        gamma=0.995,
        tau=2e-3,                 # a bit faster target update
        actor_lr=1e-4,
        critic_lr=3e-4,
        ou_sigma=0.15 * env.action_high,   # ~15% of Pmax exploration noise
        per_alpha=0.6,
        per_beta_start=0.4,
        per_beta_end=1.0,
        per_beta_steps=200_000,
        seed=0,
    )

    agent, ep_returns = train(
        env,
        episodes=args.episodes,
        steps_per_ep=args.steps_per_ep,
        cfg=ddpg,
        warmup_steps=4000,
        log_dir=args.log_dir,
    )
    print("Returns:", ep_returns)

if __name__ == "__main__":
    main()
