# train_generator_ddpg_tf.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from envs.generator_env import ISEMGeneratorEnv, PlantParams, CostParams, EnvConfig
from rl.ddpg_tf import DDPGAgent, DDPGConfig  # reuse your existing TF2 agent

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def train(
    env: ISEMGeneratorEnv,
    episodes: int = 120,
    steps_per_ep: int | None = None,
    cfg: DDPGConfig = DDPGConfig(),
    warmup_steps: int = 1_000,
    log_dir: str | Path | None = None,
):
    obs_dim = env.obs_dim
    act_low, act_high = env.action_low, env.action_high

    agent = DDPGAgent(obs_dim=obs_dim, act_dim=1, act_low=act_low, act_high=act_high, cfg=cfg)

    # logging setup
    run_dir = None
    if log_dir:
        run_dir = Path(log_dir) / f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        ensure_dir(run_dir)
        steps_csv = open(run_dir / "steps.csv", "w", newline="")
        steps_writer = csv.writer(steps_csv)
        steps_writer.writerow([
            "episode","t","reward","price","P","P_cmd","on","started",
            "revenue","var_cost","no_load","startup","ramp_cost"
        ])
        eps_rows = []
    else:
        steps_writer = None
        steps_csv = None
        eps_rows = []

    total_steps = 0
    ep_returns = []

    for ep in range(1, episodes + 1):
        o = env.reset()
        agent.reset_noise()         
        ret = 0.0
        starts = 0
        on_steps = 0
        cap_sum = 0.0

        # per-episode exploration decay can be handled inside agent if you like
        if steps_per_ep is None:
            steps_this_ep = env.cfg.episode_len
        else:
            steps_this_ep = steps_per_ep

        for t in range(steps_this_ep):
            # ALWAYS pick an action (noise is inside the agent)
            a = agent.select_action(o)  # <-- no 'add_noise' kw

            # Step env
            o2, r, done, info = env.step(a)
            ret += r
            starts += int(info["started"])
            on_steps += int(info["on"])
            cap_sum += info["P"]

            # Store transition FIRST
            agent.store((o, a, r, o2, float(done)))
            o = o2
            total_steps += 1

            # Only then update (and do 1–2 gradient steps per env step)
            update_info = {}
            if total_steps >= warmup_steps:
                # up to you; 1–2 updates per step is common for DDPG
                update_info = agent.update()
                # update_info = agent.update() or loop twice:
                # for _ in range(2): update_info = agent.update()

            if steps_writer is not None:
                steps_writer.writerow([
                    ep, t, info["profit"], info["price"], info["P"],
                    info["P_cmd"], info["on"], info["started"],
                    info["revenue"], info["var_cost"], info["no_load"],
                    info["startup"], info["ramp_cost"]
                ])

            if done:
                break


        cap_factor = cap_sum / (steps_this_ep * env.plant.P_max)
        ep_returns.append(ret)

        print(f"Episode {ep:03d} | Return {ret:,.0f} | starts={starts} | "
              f"on_steps={on_steps}/{steps_this_ep} | cap_factor={cap_factor:.2%} | last_update: {update_info}")

        if run_dir is not None:
            eps_rows.append(dict(episode=ep, episode_return=ret, starts=starts, on_steps=on_steps, cap_factor=cap_factor))

    if steps_csv is not None:
        steps_csv.close()

    if run_dir is not None:
        # save episodes.csv
        eps_df = pd.DataFrame(eps_rows)
        eps_df.to_csv(run_dir / "episodes.csv", index=False)

        # plots
        try:
            plt.figure(figsize=(7,5))
            plt.plot(range(1, len(ep_returns)+1), ep_returns, label="Return")
            plt.xlabel("Episode"); plt.ylabel("Profit (€)")
            plt.title("Generator DDPG: episode returns")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(run_dir / "ep_returns.png")
            plt.close()

            plt.figure(figsize=(7,5))
            plt.plot(eps_df["episode"], eps_df["cap_factor"])
            plt.xlabel("Episode"); plt.ylabel("Capacity factor")
            plt.title("Mean P / Pmax by episode")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(run_dir / "meanP_by_episode.png")
            plt.close()
        except Exception as e:
            print("[plot] skipped due to:", e)

        # manifest
        (run_dir / "manifest.json").write_text(
            pd.Series({
                "episodes_csv": str(run_dir / "episodes.csv"),
                "steps_csv": str(run_dir / "steps.csv"),
                "ep_returns_png": str(run_dir / "ep_returns.png"),
                "meanP_by_episode_png": str(run_dir / "meanP_by_episode.png"),
            }).to_json()
        )

    return agent, ep_returns
