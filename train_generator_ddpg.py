# train_generator_ddpg.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from envs.generator_env import ISEMGeneratorEnv
from rl.ddpg_tf import DDPGAgent, DDPGConfig

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _rollout_eval(env: ISEMGeneratorEnv, agent: DDPGAgent, episodes: int, steps_per_ep: int):
    """Evaluate the current policy with NO parameter updates, NO exploration noise."""
    returns, cap_factors = [], []
    for _ in range(episodes):
        o = env.reset()
        if hasattr(agent, "reset_noise"):
            agent.reset_noise()
        ret = 0.0
        cap_sum = 0.0
        for t in range(steps_per_ep):
            a = agent.select_action(o, explore=False)  # <- deterministic eval
            o, r, d, info = env.step(a)
            ret += r
            cap_sum += info["P"]
            if d:
                break
        cap_factor = cap_sum / (steps_per_ep * env.plant.P_max)
        returns.append(ret)
        cap_factors.append(cap_factor)
    return float(np.mean(returns)), float(np.mean(cap_factors))

def train(
    env: ISEMGeneratorEnv,
    episodes: int = 120,
    steps_per_ep: int | None = None,
    cfg: DDPGConfig = DDPGConfig(),
    warmup_steps: int = 1_000,
    log_dir: str | Path | None = None,
    # optional eval envs
    env_val: ISEMGeneratorEnv | None = None,
    env_test: ISEMGeneratorEnv | None = None,
    eval_val_eps: int = 0,
    eval_test_eps: int = 0,
    manifest_data: dict | None = None,
    eval_every: int = 10,
):
    # ---- setup ----
    try:
        obs_dim = env.obs_dim
    except AttributeError:
        obs_dim = int(np.asarray(env.reset()).shape[0])

    act_low, act_high = env.action_low, env.action_high
    agent = DDPGAgent(obs_dim=obs_dim, act_dim=1, act_low=act_low, act_high=act_high, cfg=cfg)

    # logging
    run_dir = None
    eval_writer = None
    if log_dir:
        run_dir = Path(log_dir) / f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        ensure_dir(run_dir)
        steps_csv = open(run_dir / "steps.csv", "w", newline="")
        steps_writer = csv.writer(steps_csv)
        steps_writer.writerow([
            "episode","t","profit","price","P","P_cmd","on","started",
            "revenue","var_cost","no_load","startup","ramp_cost"
        ])
        eps_rows = []
        eval_csv = open(run_dir / "eval.csv", "w", newline="")
        eval_writer = csv.writer(eval_csv)
        eval_writer.writerow(["at_episode","split","mean_return","mean_cap_factor"])
    else:
        steps_writer = None
        steps_csv = None
        eps_rows = []

    total_steps = 0
    ep_returns = []

    # ---- training loop ----
    for ep in range(1, episodes + 1):
        o = env.reset()
        if hasattr(agent, "reset_noise"):
            agent.reset_noise()
        ret = 0.0
        starts = 0
        on_steps = 0
        cap_sum = 0.0

        steps_this_ep = env.cfg.episode_len if steps_per_ep is None else steps_per_ep
        update_info = {}

        for t in range(steps_this_ep):
            # warmup with random actions in [low, high]
            if total_steps < warmup_steps:
                a = np.array([np.random.uniform(env.action_low, env.action_high)], dtype=np.float32)
            else:
                a = agent.select_action(o, explore=True)

            o2, r, done, info = env.step(a)
            ret += r
            starts += int(info["started"])
            on_steps += int(info["on"])
            cap_sum += info["P"]

            agent.store((o, a, r, o2, float(done)))
            o = o2
            total_steps += 1

            if total_steps >= warmup_steps:
                update_info = agent.update()

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
            eps_rows.append(dict(
                episode=ep,
                episode_return=ret,
                starts=starts,
                on_steps=on_steps,
                cap_factor=cap_factor
            ))

        # periodic eval
        if (ep % eval_every) == 0:
            if env_val is not None and eval_val_eps > 0:
                mret, mcap = _rollout_eval(env_val, agent, eval_val_eps, steps_this_ep)
                print(f"[EVAL] VAL  @ ep {ep}: mean_return={mret:,.0f}, mean_cap_factor={mcap:.2%}")
                if eval_writer is not None:
                    eval_writer.writerow([ep, "val", mret, mcap])

            if env_test is not None and eval_test_eps > 0:
                mret, mcap = _rollout_eval(env_test, agent, eval_test_eps, steps_this_ep)
                print(f"[EVAL] TEST @ ep {ep}: mean_return={mret:,.0f}, mean_cap_factor={mcap:.2%}")
                if eval_writer is not None:
                    eval_writer.writerow([ep, "test", mret, mcap])

    # ---- close files ----
    if log_dir:
        steps_csv.close()
        eval_csv.close()

    # ---- save logs/plots ----
    if run_dir is not None:
        eps_df = pd.DataFrame(eps_rows)
        eps_df.to_csv(run_dir / "episodes.csv", index=False)

        try:
            plt.figure(figsize=(7,5))
            plt.plot(range(1, len(ep_returns)+1), ep_returns, label="Return")
            plt.xlabel("Episode"); plt.ylabel("Profit (€)")
            plt.title("Generator DDPG: episode returns")
            plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
            plt.savefig(run_dir / "ep_returns.png"); plt.close()

            plt.figure(figsize=(7,5))
            plt.plot(eps_df["episode"], eps_df["cap_factor"])
            plt.xlabel("Episode"); plt.ylabel("Capacity factor")
            plt.title("Mean P / Pmax by episode")
            plt.grid(True, alpha=0.3); plt.tight_layout()
            plt.savefig(run_dir / "meanP_by_episode.png"); plt.close()
        except Exception as e:
            print("[plot] skipped due to:", e)

        manifest = {
            "episodes_csv": str(run_dir / "episodes.csv"),
            "steps_csv": str(run_dir / "steps.csv"),
            "eval_csv": str(run_dir / "eval.csv"),
            "ep_returns_png": str(run_dir / "ep_returns.png"),
            "meanP_by_episode_png": str(run_dir / "meanP_by_episode.png"),
        }
        if manifest_data is not None:
            manifest["meta"] = manifest_data
        (run_dir / "manifest.json").write_text(pd.Series(manifest).to_json())

    return agent, ep_returns
