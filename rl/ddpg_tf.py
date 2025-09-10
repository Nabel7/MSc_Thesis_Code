
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers

from .networks_tf import Actor, Critic
from .per_tf import PrioritizedReplayBuffer

@dataclass
class DDPGConfig:
    """
    Hyperparameters for DDPG + PER.
    """
    gamma: float = 0.99                      # discount factor
    tau: float = 5e-3                        # soft target update rate
    actor_lr: float = 1e-4                   # actor learning rate
    critic_lr: float = 1e-3                  # critic learning rate (often a bit larger than actor)
    batch_size: int = 256                    # PER batch size
    buffer_size: int = 1_000_000             # replay capacity
    per_alpha: float = 0.6                   # PER alpha (priority strength)
    per_beta_start: float = 0.4              # IS weight beta start
    per_beta_end: float = 1.0                # IS weight beta end
    per_beta_steps: int = 200_000            # steps to anneal beta
    ou_theta: float = 0.15                   # OU mean reversion
    ou_sigma: float = 0.2                    # OU volatility (units = action units)
    seed: int = 0                            # RNG seed


class OUNoise:
    """
    Ornstein–Uhlenbeck process for temporally correlated exploration.
    Good for continuous control, but the scale must match your action units.
    """
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2, seed=0):
        self.mu, self.theta, self.sigma = mu, theta, sigma
        self.rng = np.random.default_rng(seed)
        self.state = np.ones(size) * mu

    def reset(self):
        self.state[:] = self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * self.rng.standard_normal(self.state.shape)
        self.state = self.state + dx
        return self.state

class DDPGAgent:
    """
    DDPG with:
    - Deterministic policy (Actor) bounded by ScaledTanh.
    - Critic trained to TD target using target networks.
    - Prioritized Experience Replay with IS weights.
    - Soft target updates for stability.
    """

    def __init__(self, obs_dim: int, act_dim: int, act_low, act_high, cfg: DDPGConfig = DDPGConfig()):
        # Reproducibility (TF + NumPy)
        tf.random.set_seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.cfg = cfg

        # Online networks
        self.actor = Actor(obs_dim, act_dim, act_low, act_high)
        self.critic = Critic(obs_dim, act_dim)

        # Target networks (lagged copies for TD target)
        self.actor_target = Actor(obs_dim, act_dim, act_low, act_high)
        self.critic_target = Critic(obs_dim, act_dim)

        # (Recommended) Build networks once to allocate weights, then copy to targets.
        # Ensures shapes exist before set_weights (esp. with subclassed models).
        o = tf.zeros((1, obs_dim), dtype=tf.float32)
        a = tf.zeros((1, act_dim), dtype=tf.float32)
        _ = self.actor(o); _ = self.actor_target(o)
        _ = self.critic(o, a); _ = self.critic_target(o, a)

        # Initialize target networks
        self.actor_target.set_weights(self.actor.get_weights())
        self.critic_target.set_weights(self.critic.get_weights())

        # Optimizers with gradient norm clipping (extra stability)
        self.actor_opt  = optimizers.Adam(learning_rate=cfg.actor_lr,  clipnorm=1.0)
        self.critic_opt = optimizers.Adam(learning_rate=cfg.critic_lr, clipnorm=1.0)

        # Exploration noise & replay
        self.noise = OUNoise(act_dim, theta=cfg.ou_theta, sigma=cfg.ou_sigma, seed=cfg.seed)
        self.buffer = PrioritizedReplayBuffer(cfg.buffer_size, alpha=cfg.per_alpha)

        # Beta annealing schedule for IS weights
        self._beta = cfg.per_beta_start
        self._beta_step = (cfg.per_beta_end - cfg.per_beta_start) / max(1, cfg.per_beta_steps)

        # Action bounds for clipping (numpy for speed)
        self.act_low = np.array(act_low, dtype=np.float32)
        self.act_high = np.array(act_high, dtype=np.float32)

    def select_action(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        """
        Deterministic action from Actor, plus OU noise if explore=True.
        Returns a clipped action within [act_low, act_high].
        """
        obs = obs.astype(np.float32)[None, ...]
        a = self.actor(obs).numpy()[0]
        if explore:
            a = a + self.noise.sample()
        return np.clip(a, self.act_low, self.act_high)

    def store(self, transition: Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]):
        """Push one (s, a, r, s', done) into PER."""
        self.buffer.add(transition)


    def reset_noise(self):
        """Reset OU noise state at episode boundaries."""
        self.noise.reset()

    @tf.function
    def _soft_update(self, target_vars, source_vars, tau):
        """
        Polyak averaging: θ̄ ← (1−τ) θ̄ + τ θ
        Running in a @tf.function keeps this update fast on TF.
        """
        for t, s in zip(target_vars, source_vars):
            t.assign(t * (1.0 - tau) + s * tau)

    def update(self) -> Dict[str, float]:
        """
        One DDPG update step (if we have enough samples):
        1) Sample PER batch + IS weights (defensive to Nones / NaNs).
        2) Critic update (TD regression to target).
        3) Actor update (deterministic policy gradient).
        4) Update PER priorities using |TD error|.
        5) Soft-update targets.
        Returns scalars for logging, or {} if not enough clean data yet.
        """
        # Not enough experiences yet
        if len(self.buffer) < self.cfg.batch_size:
            return {}

        # Anneal importance-sampling beta toward 1.0
        self._beta = min(self.cfg.per_beta_end, self._beta + self._beta_step)

        # ---- Sample PER batch (defensive) ----
        try:
            idxs, batch, weights = self.buffer.sample(self.cfg.batch_size, beta=self._beta)
        except Exception:
            return {}

        if not batch:
            return {}

        def _finite(x) -> bool:
            try:
                arr = np.asarray(x, dtype=np.float32)
                return np.isfinite(arr).all()
            except Exception:
                return False

        # Build a clean subset aligned across idxs/batch/weights
        used_idxs, used_w = [], []
        obs_list, act_list, rew_list, nxt_list, done_list = [], [], [], [], []

        if weights is None:
            weights = [1.0] * len(batch)

        for idx, tr, w in zip(idxs, batch, weights):
            if tr is None or not isinstance(tr, (list, tuple)) or len(tr) != 5:
                continue
            o, a, r, n, d = tr
            if (o is None) or (a is None) or (n is None) or (r is None) or (d is None):
                continue
            if not (_finite(o) and _finite(a) and _finite(n) and np.isfinite(r) and np.isfinite(d)):
                continue
            used_idxs.append(idx)
            used_w.append(w)
            obs_list.append(np.asarray(o, dtype=np.float32))
            # ensure 1D scalar action per transition; will reshape to (B,1) later
            a = np.asarray(a, dtype=np.float32).reshape(-1)
            act_list.append(a[0] if a.size > 0 else np.float32(0.0))
            rew_list.append(np.float32(r))
            nxt_list.append(np.asarray(n, dtype=np.float32))
            done_list.append(np.float32(d))

        if len(used_idxs) == 0:
            return {}

        # ---- To tensors (correct shapes) ----
        obs  = tf.convert_to_tensor(np.stack(obs_list, axis=0), dtype=tf.float32)          # (B, obs_dim)
        nxt  = tf.convert_to_tensor(np.stack(nxt_list, axis=0), dtype=tf.float32)          # (B, obs_dim)
        act  = tf.convert_to_tensor(np.asarray(act_list, dtype=np.float32).reshape(-1, 1)) # (B, 1)
        rew  = tf.convert_to_tensor(np.asarray(rew_list, dtype=np.float32).reshape(-1, 1)) # (B, 1)
        done = tf.convert_to_tensor(np.asarray(done_list, dtype=np.float32).reshape(-1, 1))# (B, 1)
        w    = tf.convert_to_tensor(np.asarray(used_w, dtype=np.float32).reshape(-1, 1))   # (B, 1)

        # ---- Critic update -----------------------------------------------------
        with tf.GradientTape() as tape:
            # Target: y = r + γ (1-d) Q̄(s', μ̄(s'))
            nxt_a   = self.actor_target(nxt)                         # μ̄(s')
            q_tgt   = self.critic_target(nxt, nxt_a)                 # Q̄(s', μ̄(s'))
            y       = rew + self.cfg.gamma * (1.0 - done) * q_tgt
            y       = tf.stop_gradient(y)

            q       = self.critic(obs, act)                          # Q(s, a)

            # PER-weighted Huber loss per sample
            huber   = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)
            per_loss= huber(y, q)                                    # (B,1)
            critic_loss = tf.reduce_mean(w * per_loss)

        crit_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        crit_grads = [tf.zeros_like(v) if g is None else g
                    for g, v in zip(crit_grads, self.critic.trainable_variables)]
        clip_norm = getattr(self.cfg, "grad_clip_norm", 1.0)
        crit_grads = [tf.clip_by_norm(g, clip_norm) for g in crit_grads]
        # APPLY (this was missing in your snippet!)
        self.critic_opt.apply_gradients(zip(crit_grads, self.critic.trainable_variables))

        # ---- Actor update ------------------------------------------------------
        with tf.GradientTape() as tape:
            a_pred = self.actor(obs)                                  # μ(s)
            actor_loss = -tf.reduce_mean(self.critic(obs, a_pred))    # maximize Q ≡ minimize -Q
        act_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        act_grads = [tf.clip_by_norm(g, clip_norm) for g in act_grads]
        self.actor_opt.apply_gradients(zip(act_grads, self.actor.trainable_variables))

        # ---- PER priority updates ---------------------------------------------
        # Use |TD error| as priority; align with used_idxs only
        td_error = tf.abs(y - q)                                      # (B,1)
        td_abs   = td_error.numpy().reshape(-1)
        eps = getattr(self.cfg, "priority_epsilon", 1e-6)
        try:
            self.buffer.update_priorities(used_idxs, (td_abs + eps))
        except Exception:
            # Be tolerant if the buffer expects numpy array, etc.
            self.buffer.update_priorities(list(used_idxs), list(td_abs + eps))

        # ---- Soft update targets ----------------------------------------------
        self._soft_update(self.actor_target.variables,  self.actor.variables,  self.cfg.tau)
        self._soft_update(self.critic_target.variables, self.critic.variables, self.cfg.tau)

        return {
            "critic_loss": float(critic_loss.numpy()),
            "actor_loss":  float(actor_loss.numpy()),
            "beta":        float(self._beta),
    }
