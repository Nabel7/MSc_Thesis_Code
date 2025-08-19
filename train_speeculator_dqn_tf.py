# train_speculator_dqn_tf.py
from dataclasses import asdict
import numpy as np
import os

from rl.dqn_tf import DQNAgent, DQNConfig

# Optional CSV logging (safe fallback if utils/logger.py isn't present)
try:
    from utils.logger import CSVLogger, EpisodeLog  # your helper
except Exception:
    class EpisodeLog:
        def __init__(self, **kwargs): self.__dict__.update(kwargs)
    class CSVLogger:
        def __init__(self, episode_csv=None, step_csv=None): pass
        def log_step(self, **kwargs): pass
        def log_episode(self, *args, **kwargs): pass


def train(env_train, env_val, episodes=30, steps_per_ep=96,
          cfg: DQNConfig = DQNConfig(), warmup_steps=2000):
    """
    Train loop for discrete Δ-action DQN (dueling/double + PER).
    - Fills replay during warm-up with random actions.
    - Calls agent.update() only after warm-up AND when replay has >= batch_size items.
    - Logs per-step (optional) and per-episode summaries.
    """
    agent = DQNAgent(env_train.observation_space.shape[0], cfg)
    print("DQN config:", asdict(cfg))

    returns = []
    global_steps = 0

    # --- logging set-up (non-blocking) ---
    os.makedirs("logs", exist_ok=True)
    logger = CSVLogger(
        episode_csv="logs/spec_dqn_episodes.csv",
        step_csv="logs/spec_dqn_steps.csv"   # comment out if you don’t want per-step CSV
    )

    for ep in range(1, episodes + 1):
        obs = env_train.reset()
        ep_ret = 0.0
        clears = {0: 0, 1: 0}
        seen   = {0: 0, 1: 0}
        last_update = {}  # keep the last non-empty update logs

        # Helpful: show when training will actually start
        if ep == 1:
            print(f"[info] warmup_steps={warmup_steps}, batch_size={agent.cfg.batch_size}, "
                  f"buffer_size={agent.cfg.buffer_size}")

        for t in range(steps_per_ep):
            # ---- action selection (warm-up: random; else: epsilon-greedy policy) ----
            if global_steps < warmup_steps:
                a_idx = np.random.randint(cfg.num_bins)
                delta = agent.delta_values[a_idx]
            else:
                a_idx, delta = agent.select_action(obs, explore=True)

            # ---- environment step ----
            nxt, rew, done, info = env_train.step([delta])

            # ---- store transition ALWAYS (including warm-up) ----
            agent.store((obs, a_idx, float(rew), nxt, bool(done)))

            # ---- update ONLY when ready: after warm-up and enough samples in replay ----
            if (global_steps >= warmup_steps) and (len(agent.buffer) >= agent.cfg.batch_size):
                step_logs = agent.update() or {}
                if step_logs:
                    last_update = step_logs  # keep latest loss/eps/beta for episode summary

            # ---- optional per-step logging (robust to missing keys) ----
            try:
                logger.log_step(
                    split="train", episode=ep, t=t,
                    DAM=float(info.get("DAM", np.nan)),
                    BM=float(info.get("BM", np.nan)),
                    ref=float(info.get("ref", np.nan)),
                    side=int(info.get("side", -1)),
                    q=float(info.get("q", np.nan)),
                    delta=float(info.get("delta", np.nan)),
                    limit=float(info.get("limit", np.nan)),
                    cleared=bool(info.get("cleared", False)),
                    reward=float(rew),
                )
            except Exception:
                pass  # never let logging crash training

            # small debug print for first few steps
            if t < 5:
                try:
                    print(
                        f"[dbg] DAM={float(info.get('DAM', np.nan)):.2f} "
                        f"BM={float(info.get('BM', np.nan)):.2f} "
                        f"ref={float(info.get('ref', np.nan)):.2f} "
                        f"side={int(info.get('side', -1))} "
                        f"delta={float(info.get('delta', np.nan)):.2f} "
                        f"limit={float(info.get('limit', np.nan)):.2f} "
                        f"cleared={bool(info.get('cleared', False))} "
                        f"reward={float(rew):.3f}"
                    )
                except Exception:
                    pass

            # ---- bookkeeping ----
            ep_ret += float(rew)
            s = int(info.get("side", 0))
            seen[s] = seen.get(s, 0) + 1
            if info.get("cleared", False):
                clears[s] = clears.get(s, 0) + 1

            obs = nxt
            global_steps += 1

            if done:
                break

        # episode summary
        cr_buy  = 100.0 * (clears.get(1, 0) / max(1, seen.get(1, 0)))
        cr_sell = 100.0 * (clears.get(0, 0) / max(1, seen.get(0, 0)))
        returns.append(ep_ret)
        print(
            f"Episode {ep:03d} | Return {ep_ret:.3f} | "
            f"clear_rate(buy)={cr_buy:.2f}% clear_rate(sell)={cr_sell:.2f}% | "
            f"last_update: {last_update}"
        )

        # episode-level TRAIN log
        try:
            logger.log_episode(EpisodeLog(
                split="train",
                episode=ep,
                steps=t + 1,
                ep_return=float(ep_ret),
                clear_rate_buy=float(cr_buy),
                clear_rate_sell=float(cr_sell),
                loss=float(last_update.get("loss", np.nan)),
                eps=float(last_update.get("eps", np.nan)),
                beta=float(last_update.get("beta", np.nan)),
            ))
        except Exception:
            pass

        # ---- quick validation probe (no exploration) ----
        v_obs = env_val.reset()
        v_ret = 0.0
        for t in range(steps_per_ep):
            a_idx, delta = agent.select_action(v_obs, explore=False)
            v_obs, r, d, _ = env_val.step([delta])
            v_ret += float(r)
            if d:
                break
        print(f"  [VAL] episode return: {v_ret:.3f}")

        # episode-level VAL log
        try:
            logger.log_episode(EpisodeLog(
                split="val",
                episode=ep,
                steps=t + 1,
                ep_return=float(v_ret),
                clear_rate_buy=np.nan,
                clear_rate_sell=np.nan,
                loss=np.nan,
                eps=np.nan,
                beta=np.nan,
            ))
        except Exception:
            pass

    return agent, returns
