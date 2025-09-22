import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_spec_run(steps_path, episodes_path, output_dir="analysis_speculator"):
    steps = pd.read_csv(steps_path)
    episodes = pd.read_csv(episodes_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Episode returns
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(episodes['episode'], episodes['ep_return'], label="Episode Return")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title("Speculator Episode Returns")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "episode_returns.png")
    plt.close(fig)

    # Clear rates
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(episodes['episode'], episodes['clear_rate_buy'], label="Buy clear rate (%)")
    ax.plot(episodes['episode'], episodes['clear_rate_sell'], label="Sell clear rate (%)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Clear Rate (%)")
    ax.set_title("Buy/Sell Clear Rates Over Episodes")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "clear_rates.png")
    plt.close(fig)

    # Delta distribution
    if 'delta' in steps.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(steps['delta'].dropna(), bins=40, alpha=0.7)
        ax.set_xlabel("Delta (€/MWh)")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Delta Bids")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "delta_distribution.png")
        plt.close(fig)

    # Limit price vs DAM
    if {'limit', 'DAM'}.issubset(steps.columns):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(steps['DAM'], steps['limit'], alpha=0.4, s=10)
        ax.plot([steps['DAM'].min(), steps['DAM'].max()],
                [steps['DAM'].min(), steps['DAM'].max()],
                'r--', label='Perfect match')
        ax.set_xlabel("DAM Price (€/MWh)")
        ax.set_ylabel("Limit Price (€/MWh)")
        ax.set_title("Limit vs DAM Price")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "limit_vs_dam.png")
        plt.close(fig)

    # Reference price vs Limit
    if {'ref', 'limit'}.issubset(steps.columns):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(steps['ref'], steps['limit'], alpha=0.4, s=10)
        ax.set_xlabel("Reference Price (€/MWh)")
        ax.set_ylabel("Limit Price (€/MWh)")
        ax.set_title("Reference vs Limit Price")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "ref_vs_limit.png")
        plt.close(fig)

    print(f"[done] Analysis completed and plots saved to: {out_dir}")

# Example usage (replace these with actual paths after training):
# analyze_spec_run("logs/spec_dqn_steps.csv", "logs/spec_dqn_episodes.csv")

analyze_spec_run("logs/run_20250917-105858/spec_dqn_steps.csv", "logs/run_20250917-105858/spec_dqn_episodes.csv")