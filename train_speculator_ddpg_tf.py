
"""
Train a single-agent speculator with TensorFlow DDPG + Prioritized Replay.
Expected environment: Gym-like API.
"""
from dataclasses import asdict
import numpy as np
from rl.ddpg_tf import DDPGAgent, DDPGConfig

def train(env, episodes=200, steps_per_ep=None, cfg: DDPGConfig = DDPGConfig(),
          warmup_steps=1000, updates_per_step=1, debug_first_n=5):
    
    # Shapes and bounds
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_low = env.action_space.low
    act_high = env.action_space.high

    agent = DDPGAgent(obs_dim, act_dim, act_low, act_high, cfg=cfg)
    print("DDPG (TF) config:", asdict(cfg))
    returns = []

    # If caller doesn't specify, use env's natural episode length
    if steps_per_ep is None:
        steps_per_ep = getattr(env, "episode_len", 288)

    # Global step counter for warm-up scheduling
    gstep = 0

    for ep in range(1, episodes + 1):
        obs = env.reset()
        agent.reset_noise()

        ep_ret = 0.0
        # per-episode diagnostics
        buy_clears = sell_clears = buy_total = sell_total = 0
        deltas_buy = []
        deltas_sell = []

        last_logs = {}

        for t in range(steps_per_ep):
            # --- Action selection (random during warm-up) ---
            if gstep < warmup_steps:
                act = np.random.uniform(low=act_low, high=act_high, size=act_low.shape).astype(np.float32)
            else:
                act = agent.select_action(obs, explore=True)

            # --- Env step ---
            nxt, rew, done, info = env.step(act)
            if t < debug_first_n:
                print(
                    f"[dbg] DAM={info.get('DAM'):.2f} BM={info.get('BM'):.2f} "
                    f"ref={info.get('ref', np.nan):.2f} side={info.get('side')} "
                    f"delta={info.get('delta', np.nan):.2f} limit={info.get('limit', np.nan):.2f} "
                    f"cleared={info.get('cleared')} reward={rew:.3f}"
                )

            # --- Store transition ---
            agent.store((obs, act, float(rew), nxt, bool(done)))

            # --- Track diagnostics ---
            s = info.get("side")
            if s == 1:
                buy_total += 1
                deltas_buy.append(info.get("delta", np.nan))
                if info.get("cleared"): buy_clears += 1
            elif s == 0:
                sell_total += 1
                deltas_sell.append(info.get("delta", np.nan))
                if info.get("cleared"): sell_clears += 1

            # --- Updates (skip during warm-up) ---
            if gstep >= warmup_steps:
                for _ in range(updates_per_step):
                    logs = agent.update()
                    if logs: last_logs = logs

            # advance
            obs = nxt
            ep_ret += rew
            gstep += 1

            if done:
                break

        returns.append(ep_ret)

        # Episode summary
        cr_buy  = (buy_clears / buy_total) if buy_total else 0.0
        cr_sell = (sell_clears / sell_total) if sell_total else 0.0
        md_buy  = float(np.nanmean(deltas_buy)) if deltas_buy else float("nan")
        md_sell = float(np.nanmean(deltas_sell)) if deltas_sell else float("nan")

        print(
            f"Episode {ep:03d} | Return {ep_ret:.3f} | "
            f"clear_rate(buy)={cr_buy:.2%}, clear_rate(sell)={cr_sell:.2%} | "
            f"mean δ(buy)={md_buy:.2f}, mean δ(sell)={md_sell:.2f} | "
            f"last_update: {last_logs}"
        )

    return agent, returns

if __name__ == "__main__":
    print("Import this and call train(env, ...).")