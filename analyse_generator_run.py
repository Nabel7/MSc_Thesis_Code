# analyze_generator_run.py
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def latest_run(base: Path) -> Path | None:
    if not base.exists():
        return None
    runs = [p for p in base.iterdir() if p.is_dir() and p.name.startswith("run_")]
    if not runs:
        return None
    return max(runs, key=lambda p: p.stat().st_mtime)

def plot_episode_curves(run_dir: Path, eps: pd.DataFrame):
    # Returns
    plt.figure(figsize=(8,5))
    plt.plot(eps["episode"], eps["episode_return"], label="Train return")
    plt.xlabel("Episode"); plt.ylabel("Reward (sum)")
    plt.title("Episode Return")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(run_dir / "plot_ep_return.png"); plt.close()

    # Cap factor
    if "cap_factor" in eps.columns:
        plt.figure(figsize=(8,5))
        plt.plot(eps["episode"], eps["cap_factor"])
        plt.xlabel("Episode"); plt.ylabel("Capacity factor")
        plt.title("Capacity factor by episode")
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(run_dir / "plot_cap_factor.png"); plt.close()

    # Starts
    if "starts" in eps.columns:
        plt.figure(figsize=(8,5))
        plt.plot(eps["episode"], eps["starts"])
        plt.xlabel("Episode"); plt.ylabel("Starts")
        plt.title("Starts per episode")
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(run_dir / "plot_starts.png"); plt.close()

    # Losses (may have NaNs for early warmup)
    if "mean_actor_loss" in eps.columns and "mean_critic_loss" in eps.columns:
        plt.figure(figsize=(8,5))
        plt.plot(eps["episode"], eps["mean_actor_loss"], label="Actor loss")
        plt.plot(eps["episode"], eps["mean_critic_loss"], label="Critic loss")
        plt.xlabel("Episode"); plt.ylabel("Loss")
        plt.title("Actor/Critic Loss (per-episode mean)")
        plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
        plt.savefig(run_dir / "plot_losses.png"); plt.close()

    # Action stats
    if "mean_action" in eps.columns and "std_action" in eps.columns:
        plt.figure(figsize=(8,5))
        plt.plot(eps["episode"], eps["mean_action"], label="Mean action")
        plt.plot(eps["episode"], eps["std_action"], label="Std action")
        plt.xlabel("Episode"); plt.ylabel("MW")
        plt.title("Action statistics per episode")
        plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
        plt.savefig(run_dir / "plot_action_stats.png"); plt.close()

def plot_eval_curves(run_dir: Path, ev: pd.DataFrame):
    if ev.empty:
        return
    # Separate val vs test if present
    for split, g in ev.groupby("split"):
        plt.figure(figsize=(8,5))
        plt.plot(g["at_episode"], g["mean_return"], marker="o")
        plt.xlabel("At episode"); plt.ylabel("Mean return")
        plt.title(f"Evaluation returns ({split})")
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(run_dir / f"plot_eval_return__{split}.png"); plt.close()

        if "mean_cap_factor" in g.columns:
            plt.figure(figsize=(8,5))
            plt.plot(g["at_episode"], g["mean_cap_factor"], marker="o")
            plt.xlabel("At episode"); plt.ylabel("Mean cap factor")
            plt.title(f"Evaluation capacity factor ({split})")
            plt.grid(True, alpha=0.3); plt.tight_layout()
            plt.savefig(run_dir / f"plot_eval_capfactor__{split}.png"); plt.close()

def plot_episode_timeseries(run_dir: Path, steps: pd.DataFrame, episode: int | None = None):
    if steps.empty:
        return
    if episode is None:
        episode = int(steps["episode"].max())

    g = steps[steps["episode"] == episode].copy()
    if g.empty:
        return

    # Price vs P (two y-axes) for the chosen episode
    fig, ax1 = plt.subplots(figsize=(10,5))
    ax1.plot(g["t"], g["price"], label="Price (€/MWh)")
    ax1.set_xlabel("t (step)"); ax1.set_ylabel("€/MWh")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(g["t"], g["P"], label="P (MW)")
    ax2.set_ylabel("MW")

    plt.title(f"Episode {episode}: Price and Dispatch")
    fig.tight_layout()
    plt.savefig(run_dir / f"plot_timeseries_ep{episode}.png"); plt.close()

    # Histogram of ramps |ΔP|
    if "P" in g.columns:
        ramps = np.abs(np.diff(g["P"].to_numpy()))
        if ramps.size > 0:
            plt.figure(figsize=(8,4))
            plt.hist(ramps, bins=30)
            plt.xlabel("|ΔP| (MW)"); plt.ylabel("count")
            plt.title(f"Episode {episode}: ramp distribution")
            plt.tight_layout()
            plt.savefig(run_dir / f"plot_ramps_ep{episode}.png"); plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default="")
    parser.add_argument("--base", type=str, default="runs_generator_ddpg")
    parser.add_argument("--timeseries_ep", type=int, default=0, help="episode to visualize; 0=last")
    args = parser.parse_args()

    base = Path(args.base).resolve()
    run_dir = Path(args.run_dir).resolve() if args.run_dir else latest_run(base)
    if run_dir is None or not run_dir.exists():
        print("[error] No run_dir found."); return
    print("[analyze] run_dir:", run_dir)

    # Load CSVs if they exist
    eps = pd.read_csv(run_dir / "episodes.csv") if (run_dir / "episodes.csv").exists() else pd.DataFrame()
    ev  = pd.read_csv(run_dir / "eval.csv")     if (run_dir / "eval.csv").exists()     else pd.DataFrame()
    stp = pd.read_csv(run_dir / "steps.csv")    if (run_dir / "steps.csv").exists()    else pd.DataFrame()

    # Episode curves
    if not eps.empty:
        plot_episode_curves(run_dir, eps)

    # Eval curves
    if not ev.empty:
        plot_eval_curves(run_dir, ev)

    # One episode time series (price vs P, ramp histogram)
    if not stp.empty:
        ep = int(stp["episode"].max()) if args.timeseries_ep == 0 else int(args.timeseries_ep)
        plot_episode_timeseries(run_dir, stp, ep)

    print("[analyze] done.")

if __name__ == "__main__":
    main()
