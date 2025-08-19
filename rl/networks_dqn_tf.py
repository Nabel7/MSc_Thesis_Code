# rl/networks_dqn_tf.py
import tensorflow as tf
keras = tf.keras
layers = keras.layers

class DuelingQNetwork(keras.Model):
    """
    Q(s,·) with optional Dueling heads:
      V(s) + (A(s,a) - mean_a A(s,a))
    """
    def __init__(self, obs_dim: int, num_actions: int, hidden=(256, 256), dueling: bool = True):
        super().__init__()
        self.dueling = dueling
        self.inp = layers.InputLayer(input_shape=(obs_dim,))
        self.h = [layers.Dense(h, activation="relu") for h in hidden]
        if dueling:
            self.v = layers.Dense(1)
            self.adv = layers.Dense(num_actions)
        else:
            self.q = layers.Dense(num_actions)

    def call(self, obs, training=False):
        x = self.inp(obs)
        for layer in self.h:
            x = layer(x, training=training)
        if self.dueling:
            v = self.v(x)                    # [B, 1]
            a = self.adv(x)                  # [B, A]
            a = a - tf.reduce_mean(a, axis=1, keepdims=True)
            return v + a                     # [B, A]
        else:
            return self.q(x)                 # [B, A]
