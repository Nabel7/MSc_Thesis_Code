import tensorflow as tf
keras = tf.keras
layers = keras.layers

class ScaledTanh(layers.Layer):
    """
    Squash to [-1,1] via tanh (done in Actor.out_dense), then affine-map to [low, high].
    This guarantees the Actor’s output always respects the env’s action bounds.
    """
    def __init__(self, low, high):
        super().__init__()
        # Store bounds as tensors so graph mode works and dtypes match downstream
        self.low = tf.constant(low, dtype=tf.float32)
        self.high = tf.constant(high, dtype=tf.float32)

    def call(self, x):
        # x is in [-1,1] componentwise; cast bounds to x’s dtype for safety
        low = tf.cast(self.low, x.dtype)
        high = tf.cast(self.high, x.dtype)
        # Map: [-1,1] --( (x+1)/2 )--> [0,1] --( scale+shift )--> [low, high]
        return 0.5 * (x + 1.0) * (high - low) + low

def _mlp_layers(hidden):
    """
    Convenience: build a list of Dense layers with ReLU activations.
    hidden: tuple like (256, 256)
    """
    return [layers.Dense(h, activation="relu") for h in hidden]

class Actor(keras.Model):
    """
    Deterministic policy μθ(s): R^{obs_dim} → R^{act_dim}
    DDPG uses this to choose actions; the loss is -E_s[ Qφ(s, μθ(s)) ].
    """
    def __init__(self, obs_dim: int, act_dim: int, act_low, act_high, hidden=(256, 256)):
        super().__init__()
        # Keras 3-friendly input definition (avoids tf ops on KerasTensors)
        self.inp = layers.InputLayer(input_shape=(obs_dim,))
        # Two hidden layers with ReLU; can tune depth/width
        self.h = _mlp_layers(hidden)
        # Final tanh keeps outputs in [-1,1]; small init avoids large initial actions
        # (and helps target networks start close to online nets).
        self.out_dense = layers.Dense(
            act_dim, activation="tanh",
            kernel_initializer=keras.initializers.RandomUniform(minval=-3e-3, maxval=3e-3)
        )
        # Affine map to env action bounds [act_low, act_high]
        self.scale = ScaledTanh(act_low, act_high)

    def call(self, obs, training=False):
        """
        Forward pass. `training` flag lets BatchNorm/Dropout (if added) behave correctly.
        Shapes:
          obs: (B, obs_dim)
          return: (B, act_dim), clipped to [low, high] by ScaledTanh
        """
        x = self.inp(obs)
        for layer in self.h:
            x = layer(x, training=training)
        x = self.out_dense(x)
        return self.scale(x)


class Critic(keras.Model):
    """
    State–Action value network Qφ(s,a): R^{obs_dim+act_dim} → R
    DDPG critic loss is MSE to Bellman target:
      L = E[(r + γ Q̄(s', μ̄(s')) - Q(s,a))^2]
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256)):
        super().__init__()
        # Concatenate along the last axis; keeps shapes explicit in Keras
        self.concat = layers.Concatenate(axis=-1)
        self.inp = layers.InputLayer(input_shape=(obs_dim + act_dim,))
        self.h = _mlp_layers(hidden)
        self.out_dense = layers.Dense(1)

    def call(self, obs, act, training=False):
        """
        Forward pass.
        Shapes:
          obs: (B, obs_dim)
          act: (B, act_dim)
          return: (B, 1)
        """
        x = self.concat([obs, act])
        x = self.inp(x)
        for layer in self.h:
            x = layer(x, training=training)
        return self.out_dense(x)
