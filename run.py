#!/usr/bin/env python3
"""
Analyze a generator-DDPG training run directory.

Inputs (any may be missing; script handles gracefully):
- run_dir/steps.csv       (per-step log)
- run_dir/episodes.csv    (per-episode aggregates written by trainer)
- run_dir/eval.csv        (periodic validation/test rollouts)
- run_dir/manifest.json   (OPTIONAL: used to fetch P_max and dt_hours if present)

Outputs (written into run_dir/analysis/):
- kpis.txt                        (text summary + suggestions)
- episode_profit.png
- episode_capacity_factor.png
- profit_vs_capacity_scatter.png
- profit_vs_starts_scatter.png
- cumulative_profit.png
- per_step_profit_hist.png
- sample_episode_timeseries_[best|median|worst].png
- eval_curves.png                 (if eval.csv exists)
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------- utils ----------------------------

def _safe_read_csv(p: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(p)
    except Exception as e:
        print(f"[warn] could not read {p}: {e}")
        return None

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _get_manifest_numbers(manifest_path: Path):
    P_max = None
    dt_hours = None
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            meta = manifest.get("meta", {})
            plant = meta.get("plant", {})
            cfg = meta.get("cfg", {})
            P_max = float(plant.get("P_max")) if "P_max" in plant else None
            dt_hours = float(cfg.get("dt_hours")) if "dt_hours" in cfg else None
        except Exception as e:
            print("[warn] couldn't parse manifest.json:", e)
    return P_max, dt_hours

def _recompute_profit(df: pd.DataFrame) -> pd.Series:
    # Ensure needed columns exist
    for c in ["revenue","var_cost","no_load","startup","ramp_cost"]:
        if c not in df.columns:
            return None
    return df["revenue"] - df["var_cost"] - df["no_load"] - df["startup"] - df["ramp_cost"]

def _infer_profit(df_steps: pd.DataFrame) -> pd.Series:
    # Prefer explicit recompute, else fall back to a likely "profit-ish" column
    profit = _recompute_profit(df_steps)
    if profit is not None:
        return profit

    # common fallback: a column named "profit" or a misnamed "reward" actually holding profit
    for name in ["profit", "reward", "step_profit", "profit_eur"]:
        if name in df_steps.columns:
            return df_steps[name].astype(float)
    # Give up: zero series to avoid crashing
    print("[warn] No profit signal found; returning zeros for analysis.")
    return pd.Series(np.zeros(len(df_steps)), index=df_steps.index)

def _infer_P_max(df_steps: pd.DataFrame, manifest_Pmax: float | None) -> float | None:
    if manifest_Pmax is not None:
        return manifest_Pmax
    if "P" in df_steps.columns:
        approx = float(np.nanmax(df_steps["P"].values))
        if approx > 0:
            return approx
    return None

def _infer_dt_hours(manifest_dt: float | None) -> float:
    return float(manifest_dt) if manifest_dt is not None else 0.5  # sensible default

def _capacity_factor(ep_df: pd.DataFrame, P_max: float) -> float:
    # CF = sum(P) / (N * Pmax)
    if len(ep_df) == 0 or P_max <= 0:
        return np.nan
    return float(ep_df["P"].sum() / (len(ep_df) * P_max))

def _rolling(x, w=5):
    s = pd.Series(x)
    return s.rolling(w, min_periods=1).mean().values

# ---------------------------- main analysis ----------------------------

def analyze_run(run_dir: Path):
    out_dir = run_dir / "analysis"
    _ensure_dir(out_dir)

    # Load files
    steps = _safe_read_csv(run_dir / "steps.csv")
    for c in ["revenue","var_cost","no_load","startup","ramp_cost","P","price"]:
        if c in steps.columns:
            steps[c] = pd.to_numeric(steps[c], errors="coerce")
    episodes = _safe_read_csv(run_dir / "episodes.csv")
    evaldf = _safe_read_csv(run_dir / "eval.csv")
    P_max_mani, dt_hours_mani = _get_manifest_numbers(run_dir / "manifest.json")

    if steps is None:
        raise SystemExit("steps.csv is required for this analysis.")

    # Normalize column names a bit
    # Expected minimal set: ["episode","t","price","P","on","started", <cost cols>]
    cols = {c.lower(): c for c in steps.columns}  # map lowercase->actual
    def has(col): return col in cols
    def col(col): return cols[col]

    # Compute a clean profit series
    profit_eur = _infer_profit(steps)
    steps["profit_eur"] = profit_eur.values

    # If price or P are missing, many plots degrade gracefully
    price = steps[col("price")] if has("price") else None
    P = steps[col("P".lower())] if has("p") else None
    on = steps[col("on")] if has("on") else None
    started = steps[col("started")] if has("started") else None

    # Infer P_max, dt
    P_max = _infer_P_max(steps, P_max_mani)
    dt_hours = _infer_dt_hours(dt_hours_mani)

    # Per-episode aggregates (robust even if episodes.csv missing)
    if not has("episode"):
        raise SystemExit("steps.csv must contain an 'episode' column.")

    g = steps.groupby(steps[col("episode")])
    ep_profit = g["profit_eur"].sum()
    ep_starts = g[col("started")].sum() if has("started") else pd.Series(np.nan, index=ep_profit.index)
    ep_onsteps = g[col("on")].sum() if has("on") else pd.Series(np.nan, index=ep_profit.index)
    ep_steps = g.size()

    if P is not None and P_max is not None:
        ep_cf = g[col("p")].mean() / float(P_max)  # mean P/Pmax
    else:
        ep_cf = pd.Series(np.nan, index=ep_profit.index)

    # If episodes.csv exists, bring in episode_return as logged by trainer
    if episodes is not None:
        # Try multiple common column names
        epi = episodes.copy()
        if "episode" in epi.columns:
            epi = epi.set_index("episode")
        episode_return = None
        for c in ["episode_return", "return", "EpisodeReturn", "ep_return"]:
            if c in epi.columns:
                episode_return = epi[c]
                break
        if episode_return is not None:
            # Align indices if needed
            episode_return = episode_return.reindex(ep_profit.index)
        else:
            print("[info] episodes.csv present but no 'episode_return' column found; skipping.")
    else:
        episode_return = None

    # --- Train (episodes) vs Eval (val/test) learning curves ---
    if episode_return is not None and evaldf is not None and \
    {"at_episode","mean_return","split"}.issubset(set(evaldf.columns)):

        fig, ax = plt.subplots(figsize=(8,5))
        # Train (logged episode_return) – rolling mean to smooth noise
        trn_roll = _rolling(episode_return.values, 5)
        ax.plot(episode_return.index, trn_roll, label="train (episode_return, rolling=5)")

        # Eval (val/test) mean_return at checkpoints
        for split, grp in evaldf.groupby("split"):
            ax.plot(grp["at_episode"], grp["mean_return"],
                    marker="o", linestyle="--", label=f"{split} mean_return")

        ax.set_xlabel("Episode")
        ax.set_ylabel("Return (logged units)")
        ax.set_title("Train vs Eval returns")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "train_vs_eval_returns.png"); plt.close(fig)

        # Optional: simple generalization gap at last eval
        try:
            last_eval = evaldf.sort_values("at_episode").groupby("split").tail(1)
            last_val = float(last_eval[last_eval["split"]=="val"]["mean_return"].iloc[0])
            last_trn = float(trn_roll[min(len(trn_roll)-1, int(last_eval["at_episode"].max())-1)])
            gen_gap = last_trn - last_val

            with open(out_dir / "kpis.txt", "a") as fh:
                fh.write(f"\nGen gap (train_roll - last_val): {gen_gap:,.2f}\n")
        except Exception:
            pass


    # ---------------------------- Plots ----------------------------

    # 1) Episode profit € (and optional logged returns)
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(ep_profit.index, ep_profit.values, label="Episode profit (€)")
    ax.plot(ep_profit.index, _rolling(ep_profit.values, 5), label="Rolling mean (5)", alpha=0.8)
    if episode_return is not None:
        ax2 = ax.twinx()
        ax2.plot(episode_return.index, episode_return.values, color="tab:orange", alpha=0.6, label="Logged episode_return")
        ax2.set_ylabel("episode_return (logged units)")
        ax2.grid(False)
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")
    else:
        ax.legend(loc="best")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Profit (€)")
    ax.set_title("Episode profit")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "episode_profit.png"); plt.close(fig)

    # 2) Capacity factor per episode
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(ep_cf.index, ep_cf.values, label="Capacity factor (mean P / Pmax)")
    ax.plot(ep_cf.index, _rolling(ep_cf.values, 5), label="Rolling mean (5)", alpha=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Capacity factor")
    ax.set_title("Episode capacity factor")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "episode_capacity_factor.png"); plt.close(fig)

    # 3) Profit vs capacity factor scatter
    fig, ax = plt.subplots(figsize=(6,5))
    ax.scatter(ep_cf.values, ep_profit.values, s=20, alpha=0.7)
    ax.set_xlabel("Capacity factor")
    ax.set_ylabel("Episode profit (€)")
    ax.set_title("Profit vs capacity factor")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "profit_vs_capacity_scatter.png"); plt.close(fig)

    # 4) Profit vs starts scatter
    fig, ax = plt.subplots(figsize=(6,5))
    ax.scatter(ep_starts.values, ep_profit.values, s=20, alpha=0.7)
    ax.set_xlabel("Starts per episode")
    ax.set_ylabel("Episode profit (€)")
    ax.set_title("Profit vs starts")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "profit_vs_starts_scatter.png"); plt.close(fig)

    # 5) Cumulative profit
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(ep_profit.index, ep_profit.cumsum().values)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative profit (€)")
    ax.set_title("Cumulative profit over training")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "cumulative_profit.png"); plt.close(fig)

    # 6) Per-step profit histogram
    fig, ax = plt.subplots(figsize=(7,4))
    ax.hist(steps["profit_eur"].values, bins=60)
    ax.set_xlabel("Per-step profit (€)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of per-step profit")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "per_step_profit_hist.png"); plt.close(fig)

    # 7) Time-series for sample episodes (best / median / worst by profit)
    #    Plot P (left y), price & implied varcost rate (right y), bars for profit
    if P is not None and price is not None:
        # implied varcost rate (€/MWh) from amounts; robust to P=0
        if {"var_cost"}.issubset(set(steps.columns)):
            var_rate = steps["var_cost"].values.copy().astype(float)
            denom = np.maximum(P.values * dt_hours, 1e-6)
            var_rate = np.where(P.values > 1e-6, var_rate / denom, np.nan)
        else:
            var_rate = np.full(len(steps), np.nan)

        # helper to plot one episode
        def _plot_episode(ep_id: int, tag: str):
            sdf = steps[steps[col("episode")] == ep_id].reset_index(drop=True)
            fig, ax1 = plt.subplots(figsize=(10,4))
            ax1.plot(sdf.index, sdf["P"], label="P (MW)")
            ax1.set_ylabel("P (MW)")
            ax1.set_xlabel("Step t (within episode)")
            ax1.grid(True, alpha=0.3)

            ax2 = ax1.twinx()
            ax2.plot(sdf.index, sdf["price"], alpha=0.7, label="Price (€/MWh)")
            ax2.grid(False)
            ax2.plot(sdf.index, sdf["var_cost"], alpha=0.4, linestyle="--", label="Var cost (€)")

            # Show per-step profit as filled bars on the x-axis baseline
            ax1.fill_between(sdf.index, 0, sdf["profit_eur"], step="mid", alpha=0.2, label="step profit")

            # legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

            fig.suptitle(f"Episode {ep_id}: dispatch & economics")
            fig.tight_layout()
            fig.savefig(out_dir / f"sample_episode_timeseries_{tag}.png")
            plt.close(fig)

        # choose episodes
        order = ep_profit.sort_values()
        worst_ep = int(order.index[0])
        median_ep = int(order.index[len(order)//2])
        best_ep = int(order.index[-1])

        _plot_episode(worst_ep, "worst")
        _plot_episode(median_ep, "median")
        _plot_episode(best_ep, "best")

    # 8) Eval curves if present
    if evaldf is not None and {"at_episode","mean_return","split"}.issubset(set(evaldf.columns)):
        fig, ax = plt.subplots(figsize=(8,5))
        for split, grp in evaldf.groupby("split"):
            ax.plot(grp["at_episode"], grp["mean_return"], marker="o", label=f"{split} mean_return")
        ax.set_xlabel("Training episode")
        ax.set_ylabel("Eval mean_return (logged units)")
        ax.set_title("Periodic evaluation curves")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "eval_curves.png"); plt.close(fig)

    # ---------------------------- KPI summary + suggestions ----------------------------

    # correlations
    corr_cf = np.corrcoef(np.nan_to_num(ep_cf.values, nan=np.nanmedian(ep_cf.values)),
                          ep_profit.values)[0,1] if len(ep_profit)>1 else np.nan
    corr_starts = np.corrcoef(np.nan_to_num(ep_starts.values, nan=np.nanmedian(ep_starts.values)),
                              ep_profit.values)[0,1] if len(ep_profit)>1 else np.nan

    summary = []
    summary.append("== Generator DDPG run diagnostics ==")
    summary.append(f"Run dir: {run_dir}")
    summary.append(f"Episodes seen in steps.csv: {len(ep_profit)}")
    if P_max is not None:
        summary.append(f"P_max inferred: {P_max:.3f} MW")
    if dt_hours is not None:
        summary.append(f"dt_hours inferred: {dt_hours:.3f} h")
    summary.append(f"Total € profit (train log): {ep_profit.sum():,.2f}")
    summary.append(f"Mean € profit per episode: {ep_profit.mean():,.2f}")
    summary.append(f"Median € profit per episode: {ep_profit.median():,.2f}")
    summary.append(f"Mean capacity factor: {np.nanmean(ep_cf.values):.2%}")
    if not ep_starts.isna().all():
        summary.append(f"Mean starts/episode: {ep_starts.mean():.2f}")
    summary.append(f"Corr(profit, capacity factor): {corr_cf: .3f}")
    if not ep_starts.isna().all():
        summary.append(f"Corr(profit, starts): {corr_starts: .3f}")

    # guidance based on simple rules
    summary.append("\n== Heuristic suggestions ==")
    if P_max is not None and np.nanmean(ep_cf.values) > 0.8:
        summary.append("- Very high capacity factor on average. If this is unrealistic for your unit, consider:")
        summary.append("  • Increasing start_cost / no_load_cost, tightening min_up/down, or adding small ramp_cost_per_MW.")
    if not ep_starts.isna().all() and ep_starts.mean() > 2:
        summary.append("- Starts per episode are high. To discourage cycling, raise start_cost, min_down_steps, or ramp_cost.")
    if corr_cf < 0.1:
        summary.append("- Profit is weakly correlated with capacity factor. Try exposing stronger price/cost signals or adding recent lags.")
    if "price" in steps.columns and "var_cost" in steps.columns and P is not None:
        # crude check of average margin when running
        denom = np.maximum(P.values * dt_hours, 1e-6)
        spread = np.where(P.values > 1e-6, steps["price"].values - steps["var_cost"].values / denom, np.nan)
        avg_spread = np.nanmean(spread)
        summary.append(f"- Avg spark spread when running (approx): {avg_spread:,.2f} €/MWh")
        if avg_spread < 0:
            summary.append("  • Negative spread while running → the policy may be over-committing; increase costs/constraints or improve state features.")

    (out_dir / "kpis.txt").write_text("\n".join(summary))
    print("\n".join(summary))
    print(f"\n[done] Plots + KPIs written to: {out_dir}")

# ---------------------------- CLI ----------------------------

def _most_recent_run(base: Path) -> Path | None:
    if not base.exists():
        return None
    runs = sorted([p for p in base.glob("run_*") if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Analyze a generator-DDPG run directory.")
    ap.add_argument("--run", type=str, default=None,
                    help="Path to a run dir containing steps.csv, episodes.csv, etc. "
                         "If not provided, will try the most recent under ./runs_generator_ddpg.")
    args = ap.parse_args()

    if args.run is None:
        default_base = Path("./runs_generator_ddpg")
        guess = _most_recent_run(default_base)
        if guess is None:
            raise SystemExit("No --run provided and no runs found under ./runs_generator_ddpg")
        run_dir = guess
    else:
        run_dir = Path(args.run)

    analyze_run(run_dir)
