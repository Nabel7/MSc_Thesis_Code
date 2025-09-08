# rl/dqn_tf.py
from dataclasses import dataclass
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from .per_tf import PrioritizedReplayBuffer

@dataclass
class DQNConfig:
    num_bins: int = 41
    delta_max: float = 50.0
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 128
    buffer_size: int = 100_000
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_beta_steps: int = 200_000
    dueling: bool = True
    double: bool = True
    tau: float = 5e-3
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 50_000
    seed: int = 0
    loss: str = "huber"
    grad_clip_norm: float = 10.0
    reward_transform: str = "none"          # applied in the train loop
    n_step: int = 1
    mix_nstep_lambda: float = 0.0
    priority_epsilon: float = 1e-6
    
    def __post_init__(self):
        assert 0 <= self.eps_end <= self.eps_start <= 1
        assert 0 <= self.per_alpha <= 1
        assert 0 <= self.per_beta_start <= 1 and 0 <= self.per_beta_end <= 1
        assert self.n_step >= 1
        assert 0.0 <= self.mix_nstep_lambda <= 1.0

class QNet(Model):
    def __init__(self, obs_dim, num_actions, dueling=True):
        super().__init__()
        self.dueling = dueling
        self.fc1 = layers.Dense(256, activation="relu")
        self.fc2 = layers.Dense(256, activation="relu")
        if dueling:
            self.v = layers.Dense(1)
            self.a = layers.Dense(num_actions)
        else:
            self.out = layers.Dense(num_actions)

    def call(self, x, training=False):
        x = self.fc1(x); x = self.fc2(x)
        if self.dueling:
            v = self.v(x); a = self.a(x)
            a_mean = tf.reduce_mean(a, axis=1, keepdims=True)
            return v + (a - a_mean)
        return self.out(x)

class DQNAgent:
    def __init__(self, obs_dim: int, cfg: DQNConfig):
        tf.random.set_seed(cfg.seed); np.random.seed(cfg.seed)
        self.cfg = cfg
        self.num_actions = cfg.num_bins
        self.delta_values = np.linspace(-cfg.delta_max, cfg.delta_max, cfg.num_bins).astype(np.float32)

        self.online = QNet(obs_dim, self.num_actions, dueling=cfg.dueling)
        self.target = QNet(obs_dim, self.num_actions, dueling=cfg.dueling)
        dummy = tf.zeros((1, obs_dim), dtype=tf.float32)
        _ = self.online(dummy); _ = self.target(dummy)
        self.target.set_weights(self.online.get_weights())

        self.opt = optimizers.Adam(cfg.lr)
        self.buffer = PrioritizedReplayBuffer(cfg.buffer_size, alpha=cfg.per_alpha)

        self._beta = cfg.per_beta_start
        self._beta_step = (cfg.per_beta_end - cfg.per_beta_start) / max(1, cfg.per_beta_steps)
        self.eps = cfg.eps_start
        self.eps_step = (cfg.eps_start - cfg.eps_end) / max(1, cfg.eps_decay_steps)

    def select_action(self, obs: np.ndarray, explore=True):
        if explore and (np.random.rand() < self.eps):
            a_idx = np.random.randint(self.num_actions)
            a_delta = float(self.delta_values[a_idx])
            self.eps = max(self.cfg.eps_end, self.eps - self.eps_step)
            return a_idx, a_delta
        q = self.online(obs.astype(np.float32)[None, ...]).numpy()[0]
        a_idx = int(np.argmax(q)); a_delta = float(self.delta_values[a_idx])
        if explore:
            self.eps = max(self.cfg.eps_end, self.eps - self.eps_step)
        return a_idx, a_delta

    def store(self, transition):
        self.buffer.add(transition)

    @tf.function
    def _soft_update(self, tgt_vars, src_vars, tau):
        for t, s in zip(tgt_vars, src_vars):
            t.assign(t * (1.0 - tau) + s * tau)

    def update(self):
        if len(self.buffer) < self.cfg.batch_size:
            return {}
        self._beta = min(self.cfg.per_beta_end, self._beta + self._beta_step)
        idxs, batch, weights = self.buffer.sample(self.cfg.batch_size, beta=self._beta)
        if not batch or any(b is None for b in batch):
            return {}

        obs, a_idx, rew, nxt, done = zip(*batch)
        obs  = tf.convert_to_tensor(np.array(obs), dtype=tf.float32)
        a_idx= tf.convert_to_tensor(np.array(a_idx), dtype=tf.int32)[:, None]
        rew  = tf.convert_to_tensor(np.array(rew), dtype=tf.float32)[:, None]
        nxt  = tf.convert_to_tensor(np.array(nxt), dtype=tf.float32)
        done = tf.convert_to_tensor(np.array(done, dtype=np.float32), dtype=tf.float32)[:, None]
        w    = tf.convert_to_tensor(np.array(weights), dtype=tf.float32)[:, None]

        with tf.GradientTape() as tape:
            q_online = self.online(obs)
            q_sa = tf.gather(q_online, a_idx, batch_dims=1)

            if self.cfg.double:
                a_star = tf.argmax(self.online(nxt), axis=1)
                q_tgt_all = self.target(nxt)
                q_next = tf.gather(q_tgt_all, a_star[:, None], batch_dims=1)
            else:
                q_next = tf.reduce_max(self.target(nxt), axis=1, keepdims=True)

            y = rew + self.cfg.gamma * (1.0 - done) * q_next
            td_err = y - q_sa
            loss = tf.reduce_mean(w * tf.keras.losses.huber(y_true=y, y_pred=q_sa))

        grads = tape.gradient(loss, self.online.trainable_variables)
        grads = [tf.clip_by_norm(g, 10.0) for g in grads]
        self.opt.apply_gradients(zip(grads, self.online.trainable_variables))
        self.buffer.update_priorities(idxs, (tf.abs(td_err).numpy().squeeze(-1) + 1e-6))
        self._soft_update(self.target.variables, self.online.variables, self.cfg.tau)

        return {"loss": float(loss.numpy()), "eps": float(self.eps), "beta": float(self._beta)}
