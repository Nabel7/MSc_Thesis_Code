
from dataclasses import asdict
from rl.maddpg_tf import MADDPG, MADDPGConfig

def train(env, episodes=200, steps_per_ep=288, cfg: MADDPGConfig = MADDPGConfig()):
    """
    Parameters
    ----------
    env : SpeculatorEnv
        Your environment (state, action, reward).
    episodes : int
        Number of episodes to roll out. Each episode samples a *contiguous* window.
    steps_per_ep : int
        Steps per episode (should match env.episode_len for full windows, but can be smaller).
    cfg : DDPGConfig 
        Hyperparameters for the DDPG agent (+ PER settings).

    Returns
    -------
    agent : DDPGAgent
        Trained agent (Actor/Critic + targets + replay buffer).
    returns : list[float]
        Sum of rewards per episode (already scaled by env.reward_scale).
    """

    # 1) Build the multi-agent system from env shapes/bounds.
    #    Each agent i has its own actor π_i and critic Q_i(s_all, a_all).
    maddpg = MADDPG(env.obs_dims, env.act_dims, env.act_lows, env.act_highs, cfg=cfg)
    print("MADDPG (TF) config:", asdict(cfg))
    for ep in range(1, episodes + 1):
        # 2) Reset the environment -> list of per-agent observations [o_1, ..., o_N]
        obs_list = env.reset()

        # Running per-episode return per agent (scalar sum of rewards per agent)
        ep_rets = [0.0] * len(env.obs_dims)
        for t in range(steps_per_ep):
            # 3) Select decentralized actions π_i(o_i) + exploration (per agent).
            #    acts is a list: [a_1, ..., a_N], where each a_i matches agent i's action space.
            acts = maddpg.select_actions(obs_list, explore=True)

            # 4) Environment transition with joint action:
            #    nxt_list: list of next observations per agent
            #    rews    : list/array of rewards per agent [r_1, ..., r_N]
            #    done    : bool episode-termination flag
            #    info    : dict for debugging/metrics
            nxt_list, rews, done, info = env.step(acts)

            # 5) Build centralized state/action for critics:
            #    concat_obs:  [o_1, ..., o_N] -> s_all
            #    concat_act:  [a_1, ..., a_N] -> a_all
            obs_all = env.concat_obs(obs_list)
            act_all = env.concat_act(acts)
            nxt_all = env.concat_obs(nxt_list)

            # 6) Store one transition in the shared buffer (maddpg handles splitting across agents):
            #    (s_all, a_all, r_vector, s_all', done)
            # NOTE: r_vector should be the full vector [r_1,...,r_N]; each critic uses its own r_i.
            maddpg.store((obs_all, act_all, rews, nxt_all, bool(done)))

             # 7) One learning step:
            #    Internally, MADDPG typically:
            #      - samples a batch of (s_all, a_all, r, s'_all, done)
            #      - for each agent i:
            #          * updates Q_i with TD target using target nets and joint (s_all', a'_all)
            #          * updates π_i to maximize Q_i(s_all, [a_1,...,π_i(o_i),...,a_N])
            logs = maddpg.update()

             # 8) Advance the trajectory
            obs_list = nxt_list
            ep_rets = [a + float(b) for a, b in zip(ep_rets, rews)]
            if done:
                break
        print(f"Episode {ep:03d} | Returns {ep_rets} | last_update: {logs}")
    return maddpg

if __name__ == "__main__":
    print("Import and call train(env, ...).")
