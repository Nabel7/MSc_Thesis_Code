
"""
Lightweight MADDPG (TF) scaffold with shared PER and centralized critics.
NOTE: This is a scaffold; you'll need to adapt env slicing utilities as with the PyTorch version.
"""
from dataclasses import dataclass, asdict
from typing import List, Dict
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers

from .networks_tf import Actor, Critic
from .per_tf import PrioritizedReplayBuffer

# -----------------------------
# Hyperparameters for MADDPG
# -----------------------------
@dataclass
class MADDPGConfig:
    gamma: float = 0.99          # discount factor
    tau: float = 5e-3            # Polyak averaging rate for target nets
    actor_lr: float = 1e-4       # actor learning rate
    critic_lr: float = 1e-3      # critic learning rate
    batch_size: int = 256        # PER batch size
    buffer_size: int = 1_000_000 # replay capacity
    per_alpha: float = 0.6       # PER priority exponent (0=>uniform,1=>full PER)
    per_beta_start: float = 0.4  # IS weight exponent start
    per_beta_end: float = 1.0    # IS weight exponent end
    per_beta_steps: int = 200_000# steps to anneal beta linearly



class MADDPG:
    """
    Multi-Agent DDPG with:
      • Decentralized actors: π_i(o_i)
      • Centralized critics:  Q_i(s_all, a_all)
      • Shared Prioritized Replay
      • Soft target updates
    """
    def __init__(self, obs_dims: List[int], act_dims: List[int], act_lows, act_highs, cfg: MADDPGConfig = MADDPGConfig()):
        self.cfg = cfg
        self.n = len(obs_dims) 
        self.obs_dims = obs_dims
        self.act_dims = act_dims
        # (act_lows / act_highs) are lists of per-agent action bounds (np arrays)

        # --------- Actors (online + target), one per agent ----------
        self.actors = [Actor(obs_dims[i], act_dims[i], act_lows[i], act_highs[i]) for i in range(self.n)]
        self.target_actors = [Actor(obs_dims[i], act_dims[i], act_lows[i], act_highs[i]) for i in range(self.n)]

        # (Important) Build once to create variables, then copy weights to targets.
        for i in range(self.n):
            self.target_actors[i].set_weights(self.actors[i].get_weights())


         # --------- Centralized critics (online + target), one per agent ----------
        obs_all_dim = sum(obs_dims)
        act_all_dim = sum(act_dims)
        self.critics = [Critic(obs_all_dim, act_all_dim) for _ in range(self.n)]
        self.target_critics = [Critic(obs_all_dim, act_all_dim) for _ in range(self.n)]
        for i in range(self.n):
            self.target_critics[i].set_weights(self.critics[i].get_weights())


        # --------- Optimizers (one per actor/critic) ----------
        # TODO (optional): add clipnorm=1.0 for a bit more stability on noisy gradients.
        self.actor_opts = [optimizers.Adam(cfg.actor_lr) for _ in range(self.n)]
        self.critic_opts = [optimizers.Adam(cfg.critic_lr) for _ in range(self.n)]

        # --------- Shared PER & beta schedule ----------
        self.buffer = PrioritizedReplayBuffer(cfg.buffer_size, alpha=cfg.per_alpha)
        self._beta = cfg.per_beta_start
        self._beta_step = (cfg.per_beta_end - cfg.per_beta_start) / max(1, cfg.per_beta_steps)

    def select_actions(self, obs_list: List[np.ndarray], explore=True, noise_std=0.2):
        """
        Decentralized action selection.
        obs_list: [o_1, ..., o_n], each o_i shape (obs_dim_i,)
        Returns:  [a_1, ..., a_n], each a_i shape (act_dim_i,)
        """
        acts = []
        for i in range(self.n):
            o = obs_list[i].astype(np.float32)[None, ...]
            a = self.actors[i](o).numpy()[0]
            if explore:
                # Gaussian exploration; scale must match action units (€/MWh if absolute).
                # For delta-actions around DAM, consider larger noise (e.g. 5–10).
                a = a + np.random.randn(*a.shape) * noise_std
            acts.append(a)
        return acts

    @tf.function
    def _soft_update(self, target_vars, source_vars, tau):
        """Polyak averaging: θ̄ ← (1−τ) θ̄ + τ θ (run under TF graph for speed)."""
        for t, s in zip(target_vars, source_vars):
            t.assign(t * (1.0 - tau) + s * tau)

    def store(self, transition):
        """
        Push one centralized transition into PER:
          transition = (obs_all, act_all, rewards_list, next_obs_all, done)
        where:
          obs_all/next_obs_all: concatenated across agents (shape: (Σobs,))
          act_all:              concatenated across agents (shape: (Σact,))
          rewards_list:         list length n (per-agent rewards)
          done:                 bool
        """
        self.buffer.add(transition)

    def update(self) -> Dict[str, float]:
        """
        One learning step if we have enough samples:
          1) Sample PER batch (+ IS weights)
          2) Build target joint action a'_all = [π̄_1(o'_1), ..., π̄_n(o'_n)]
          3) For each agent i:
               - Critic_i TD regression to y_i = r_i + γ(1-d) Q̄_i(s'_all, a'_all)
               - Actor_i gradient step to maximize Q_i(s_all, [a_1,...,π_i(o_i),...,a_n])
          4) Update PER priorities using aggregate TD error per sample
          5) Soft-update all targets
        Returns a dict of scalar logs, or {} if the buffer isn’t warm yet.
        """
        if len(self.buffer) < self.cfg.batch_size:
            return {}
        
        # Anneal beta toward 1 (unbiased IS weights as training progresses)
        self._beta = min(self.cfg.per_beta_end, self._beta + self._beta_step)

        # ----- 1) Sample from PER -------------------------------------------------------------
        idxs, batch, weights = self.buffer.sample(self.cfg.batch_size, beta=self._beta)
        obs_all, act_all, rewards_list, next_obs_all, done = zip(*batch)

        # Convert to tensors (B = batch_size)
        obs_all = tf.convert_to_tensor(np.array(obs_all), dtype=tf.float32)
        act_all = tf.convert_to_tensor(np.array(act_all), dtype=tf.float32)
        next_obs_all = tf.convert_to_tensor(np.array(next_obs_all), dtype=tf.float32)
        done = tf.convert_to_tensor(np.array(done, dtype=np.float32), dtype=tf.float32)[:, None]
        w = tf.convert_to_tensor(np.array(weights), dtype=tf.float32)[:, None]

        # ----- 2) Build target joint action using target actors -------------------------------
        # Recompute per-agent slices over the concatenated observation vector.
        # TODO (optional): precompute slices once in __init__ to avoid doing it every update.

        a_next_parts = []
        start = 0
        obs_slices = []
        for i in range(self.n):
            end = start + self.obs_dims[i]
            obs_slices.append((start, end))
            start = end
        for i in range(self.n):
            s, e = obs_slices[i]
            o_i = next_obs_all[:, s:e]
            a_i = self.target_actors[i](o_i)
            a_next_parts.append(a_i)
        a_next_all = tf.concat(a_next_parts, axis=-1)

        losses = {}
        td_errors = []

        # ----- 3a) Critic updates (one critic per agent) -------------------------------------
        for i in range(self.n):
            # Build per-agent reward vector r_i (B,1)
            r_i = tf.convert_to_tensor(np.array([r[i] for r in rewards_list]), dtype=tf.float32)[:, None]
            with tf.GradientTape() as tape:
                q_targ = self.target_critics[i](next_obs_all, a_next_all)
                y = r_i + self.cfg.gamma * (1.0 - done) * q_targ
                q = self.critics[i](obs_all, act_all)
                td = tf.abs(y - q)
                loss = tf.reduce_mean(w * tf.square(y - q))
            grads = tape.gradient(loss, self.critics[i].trainable_variables)
            grads = [tf.clip_by_norm(g, 1.0) for g in grads]
            self.critic_opts[i].apply_gradients(zip(grads, self.critics[i].trainable_variables))
            losses[f"critic_loss_{i}"] = float(loss.numpy())
            td_errors.append(td)

        # ----- 3b) Actor updates (one actor per agent) ---------------------------------------
        # Build current joint action a_all = [π_1(o_1),...,π_n(o_n)]
        a_cur_parts = []
        start = 0
        for i in range(self.n):
            s, e = obs_slices[i]
            o_i = obs_all[:, s:e]
            a_i = self.actors[i](o_i)
            a_cur_parts.append(a_i)
        a_cur_all = tf.concat(a_cur_parts, axis=-1)

        for i in range(self.n):
            with tf.GradientTape() as tape:
                # Optional micro-optimization:
                # you can stop_gradient other agents' actions to avoid unnecessary backprop:
                # a_list = [a_cur_parts[j] if j == i else tf.stop_gradient(a_cur_parts[j]) for j in range(self.n)]
                # a_all_for_i = tf.concat(a_list, axis=-1)
                # loss = -tf.reduce_mean(self.critics[i](obs_all, a_all_for_i))
                loss = -tf.reduce_mean(self.critics[i](obs_all, a_cur_all))
            grads = tape.gradient(loss, self.actors[i].trainable_variables)
            grads = [tf.clip_by_norm(g, 1.0) for g in grads]
            self.actor_opts[i].apply_gradients(zip(grads, self.actors[i].trainable_variables))
            losses[f"actor_loss_{i}"] = float(loss.numpy())

        # ----- 4) PER priorities: aggregate TD error across agents ----------------------------
        # Here we use max across agents; alternatives are mean or sum (domain choice).
        td_max = tf.reduce_max(tf.concat(td_errors, axis=1), axis=1) + 1e-6
        self.buffer.update_priorities(idxs, td_max.numpy())

        # ----- 5) Soft updates for all target networks ----------------------------------------
        for i in range(self.n):
            self._soft_update(self.target_actors[i].variables, self.actors[i].variables, self.cfg.tau)
            self._soft_update(self.target_critics[i].variables, self.critics[i].variables, self.cfg.tau)
        losses["beta"] = float(self._beta)
        return losses
