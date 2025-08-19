
# plots/diagnostics.py
# Read CSV logs and produce diagnostic figures.
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def _savefig(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_episode_returns(ep_csv: Path, outdir: Path, window: int = 10):
    df = pd.read_csv(ep_csv)
    df["episode"] = df["episode"].astype(int)
    fig = plt.figure()
    for split in ["train","val"]:
        d = df[df["split"]==split].sort_values("episode")
        if len(d)==0:
            continue
        y = d["ep_return"].rolling(window=window, min_periods=1).mean()
        plt.plot(d["episode"], y, label=f"{split} (rolling {window})")
    plt.xlabel("Episode")
    plt.ylabel("Return (rolling mean)")
    plt.legend()
    _savefig(outdir / "episode_return_train_vs_val.png")

def plot_loss_eps_beta(ep_csv: Path, outdir: Path, window: int = 10):
    df = pd.read_csv(ep_csv)
    d = df[df["split"]=="train"].sort_values("episode")
    if len(d)==0:
        return
    # Critic (or DQN) loss
    fig = plt.figure()
    plt.plot(d["episode"], d["loss"].rolling(window, min_periods=1).mean())
    plt.xlabel("Episode"); plt.ylabel("Loss (rolling)")
    _savefig(outdir / "loss_train.png")

    # Epsilon
    fig = plt.figure()
    plt.plot(d["episode"], d["eps"].rolling(window, min_periods=1).mean())
    plt.xlabel("Episode"); plt.ylabel("Epsilon (ε)")
    _savefig(outdir / "epsilon_schedule.png")

    # Beta
    fig = plt.figure()
    plt.plot(d["episode"], d["beta"].rolling(window, min_periods=1).mean())
    plt.xlabel("Episode"); plt.ylabel("PER beta (β)")
    _savefig(outdir / "per_beta_schedule.png")

def plot_clear_rates(ep_csv: Path, outdir: Path, window: int = 10):
    df = pd.read_csv(ep_csv)
    d = df[df["split"]=="train"].sort_values("episode")
    if len(d)==0:
        return
    fig = plt.figure()
    plt.plot(d["episode"], d["clear_rate_buy"].rolling(window, min_periods=1).mean(), label="buy")
    plt.plot(d["episode"], d["clear_rate_sell"].rolling(window, min_periods=1).mean(), label="sell")
    plt.xlabel("Episode"); plt.ylabel("Clear rate (rolling)")
    plt.legend()
    _savefig(outdir / "clear_rates_train.png")

def plot_action_histograms(step_csv: Path, outdir: Path):
    if not step_csv.exists():
        return
    df = pd.read_csv(step_csv)
    for split in ["train","val"]:
        d = df[df["split"]==split]
        if len(d)==0:
            continue
        for side in [0,1]:
            s = d[d["side"]==side]
            if len(s)==0:
                continue
            fig = plt.figure()
            s["delta"].plot(kind="hist", bins=41)
            plt.xlabel(f"Delta (€/MWh) | split={split} side={'sell(0)' if side==0 else 'buy(1)'}")
            plt.ylabel("Count")
            _savefig(outdir / f"hist_delta_{split}_side{side}.png")

def plot_clear_vs_delta(step_csv: Path, outdir: Path, bins: int = 21):
    if not step_csv.exists():
        return
    df = pd.read_csv(step_csv)
    for split in ["train","val"]:
        d = df[df["split"]==split]
        if len(d)==0:
            continue
        for side in [0,1]:
            s = d[d["side"]==side]
            if len(s)==0:
                continue
            # Bin by delta
            cats = pd.qcut(s["delta"], q=min(bins, max(2, s["delta"].nunique())), duplicates="drop")
            grp = s.groupby(cats)
            clear_rate = grp["cleared"].mean()
            mean_reward = grp["reward"].mean()
            centers = grp["delta"].mean()

            # Clear rate
            fig = plt.figure()
            plt.plot(centers.values, clear_rate.values)
            plt.xlabel("Delta bin center (€/MWh)")
            plt.ylabel("Clear rate")
            _savefig(outdir / f"clear_rate_vs_delta_{split}_side{side}.png")

            # Mean reward
            fig = plt.figure()
            plt.plot(centers.values, mean_reward.values)
            plt.xlabel("Delta bin center (€/MWh)")
            plt.ylabel("Mean reward")
            _savefig(outdir / f"mean_reward_vs_delta_{split}_side{side}.png")

def plot_distribution_shift(step_csv: Path, outdir: Path):
    if not step_csv.exists():
        return
    df = pd.read_csv(step_csv)
    for col in ["DAM","BM","ref"]:
        fig = plt.figure()
        for split in ["train","val"]:
            d = df[df["split"]==split]
            if len(d)==0:
                continue
            d[col].plot(kind="hist", bins=50, alpha=0.5)
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.legend(["train","val"])
        _savefig(outdir / f"dist_{col}_train_vs_val.png")
    # Spread distributions
    df["DAM_minus_ref"] = df["DAM"] - df["ref"]
    df["BM_minus_DAM"] = df["BM"] - df["DAM"]
    for col in ["DAM_minus_ref","BM_minus_DAM"]:
        fig = plt.figure()
        for split in ["train","val"]:
            d = df[df["split"]==split]
            if len(d)==0:
                continue
            d[col].plot(kind="hist", bins=60, alpha=0.5)
        plt.xlabel(col); plt.ylabel("Count")
        plt.legend(["train","val"])
        _savefig(outdir / f"dist_{col}_train_vs_val.png")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode_csv", required=True, help="Path to episode-level CSV (from CSVLogger).")
    ap.add_argument("--step_csv", default="", help="Optional path to step-level CSV.")
    ap.add_argument("--out", default="reports", help="Output directory for figures.")
    ap.add_argument("--window", type=int, default=10, help="Rolling window for smoothing.")
    args = ap.parse_args()

    ep_csv = Path(args.episode_csv)
    st_csv = Path(args.step_csv) if args.step_csv else Path("")
    outdir = Path(args.out)

    plot_episode_returns(ep_csv, outdir, window=args.window)
    plot_loss_eps_beta(ep_csv, outdir, window=args.window)
    plot_clear_rates(ep_csv, outdir, window=args.window)
    if st_csv:
        plot_action_histograms(st_csv, outdir)
        plot_clear_vs_delta(st_csv, outdir)
        plot_distribution_shift(st_csv, outdir)

if __name__ == "__main__":
    main()
