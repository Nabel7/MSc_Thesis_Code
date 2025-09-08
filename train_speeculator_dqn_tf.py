# train_speeculator_dqn_tf.py
from dataclasses import asdict
import numpy as np
from typing import Optional
from rl.dqn_tf import DQNAgent, DQNConfig
from utils.logger import CSVLogger, EpisodeLog

def train(env_train, env_val,
          episodes: int = 30, steps_per_ep: int = 96,
          cfg: DQNConfig = DQNConfig(),
          warmup_steps: int = 2000,
          logger: Optional[CSVLogger] = None,
          run_id: str = ""):
    """
    Discrete-Δ DQN training loop (dueling/double, PER).
    - Warmup with random actions then epsilon-greedy via agent.select_action.
    - Update only when replay has >= batch_size.
    - Logs per-step (optional) + per-episode summaries compatible with CSVLogger.
    """
    agent = DQNAgent(env_train.observation_space.shape[0], cfg)
    print("DQN config:", asdict(cfg))

    if logger is None:
        logger = CSVLogger("logs/spec_dqn_episodes.csv", "logs/spec_dqn_steps.csv")

    returns = []
    global_steps = 0

    print(f"[info] warmup_steps={warmup_steps}, batch_size={cfg.batch_size}, buffer_size={cfg.buffer_size}")

    for ep in range(1, episodes + 1):
        obs = env_train.reset()
        ep_ret = 0.0
        clears = {0: 0, 1: 0}
        seen   = {0: 0, 1: 0}
        last_update = {}

        for t in range(steps_per_ep):
            # action
            if global_steps < warmup_steps:
                a_idx = np.random.randint(cfg.num_bins)
                delta = float(np.linspace(-cfg.delta_max, cfg.delta_max, cfg.num_bins, dtype=np.float32)[a_idx])
            else:
                a_idx, delta = agent.select_action(obs, explore=True)

            nxt, rew, done, info = env_train.step([delta])

            # store & maybe update
            agent.store((obs, a_idx, float(rew), nxt, bool(done)))
            if (global_steps >= warmup_steps) and (len(agent.buffer) >= cfg.batch_size):
                upd = agent.update() or {}
                if upd: last_update = upd

            # per-step logging
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
                pass

            # book-keeping
            ep_ret += float(rew)
            s = int(info.get("side", 0))
            seen[s] = seen.get(s, 0) + 1
            if info.get("cleared", False):
                clears[s] = clears.get(s, 0) + 1

            obs = nxt
            global_steps += 1
            if done:
                break

        cr_buy  = 100.0 * (clears.get(1, 0) / max(1, seen.get(1, 0)))
        cr_sell = 100.0 * (clears.get(0, 0) / max(1, seen.get(0, 0)))
        returns.append(ep_ret)

        print(
            f"Episode {ep:03d} | Return {ep_ret:.3f} | "
            f"clear_rate(buy)={cr_buy:.2f}% clear_rate(sell)={cr_sell:.2f}% | "
            f"last_update: {last_update}"
        )

        # episode TRAIN CSV
        try:
            logger.log_episode(EpisodeLog(
                split="train", episode=ep, steps=t + 1,
                ep_return=float(ep_ret),
                clear_rate_buy=float(cr_buy),
                clear_rate_sell=float(cr_sell),
                loss=float(last_update.get("loss", np.nan)),
                eps=float(last_update.get("eps", np.nan)),
                beta=float(last_update.get("beta", np.nan)),
            ))
        except Exception:
            pass

        # quick VAL (greedy)
        v_obs = env_val.reset()
        v_ret = 0.0
        for t in range(steps_per_ep):
            a_idx, delta = agent.select_action(v_obs, explore=False)
            v_obs, r, d, v_info = env_val.step([delta])
            v_ret += float(r)
            try:
                logger.log_step(
                    split="val", episode=ep, t=t,
                    DAM=float(v_info.get("DAM", np.nan)),
                    BM=float(v_info.get("BM", np.nan)),
                    ref=float(v_info.get("ref", np.nan)),
                    side=int(v_info.get("side", -1)),
                    q=float(v_info.get("q", np.nan)),
                    delta=float(v_info.get("delta", np.nan)),
                    limit=float(v_info.get("limit", np.nan)),
                    cleared=bool(v_info.get("cleared", False)),
                    reward=float(r),
                )
            except Exception:
                pass
            if d: break
        print(f"  [VAL] episode return: {v_ret:.3f}")

        try:
            logger.log_episode(EpisodeLog(
                split="val", episode=ep, steps=t + 1,
                ep_return=float(v_ret),
                clear_rate_buy=np.nan,
                clear_rate_sell=np.nan,
                loss=np.nan, eps=np.nan, beta=np.nan,
            ))
        except Exception:
            pass

    return agent, returns
