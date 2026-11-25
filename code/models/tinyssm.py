"""
TinySSM: 1x1 conv mixes the 6 axes -> residual blocks of diagonal SSM + 1x1 conv -> GAP -> softmax. 378 params.
"""
import keras
import numpy as np


@keras.saving.register_keras_serializable()
class DiagonalSSM(keras.layers.Layer):
    """S4D-style real diagonal linear state space model, one SSM with `state_size` states per channel.
    x_t = A_bar x_{t-1} + B_bar u_t,  y_t = C x_t + D u_t,  A_bar = exp(dt A),  B_bar = (A_bar - 1) / A (ZOH, B = 1).
    Trained in convolution form y = K * u with K[l] = sum_n C_n B_bar_n A_bar_n^l; runs as the recurrence on an MCU."""

    def __init__(self, state_size=4, **kwargs):
        super().__init__(**kwargs)
        self.state_size = state_size

    def build(self, input_shape):
        self.length, self.channels = input_shape[1], input_shape[2]
        n = self.state_size
        # S4D-Real init: A_n = -(n + 1); dt log-uniform in [0.02, 0.2] -> memory of ~1-50 rows
        self.log_neg_a = self.add_weight(shape=(self.channels, n), initializer=keras.initializers.Constant(
            np.log(np.tile(np.arange(1, n + 1, dtype=np.float32), (self.channels, 1)))))
        self.log_dt = self.add_weight(shape=(self.channels,), initializer=keras.initializers.RandomUniform(
            np.log(0.02), np.log(0.2)))
        self.c = self.add_weight(shape=(self.channels, n), initializer=keras.initializers.RandomNormal(stddev=0.5))
        self.d = self.add_weight(shape=(self.channels,), initializer="ones")

    def call(self, u):
        ops = keras.ops
        a = -ops.exp(self.log_neg_a)                                              # [H, N]
        dt_a = ops.exp(self.log_dt)[:, None] * a
        b_bar = (ops.exp(dt_a) - 1.0) / a
        pos = ops.arange(self.length, dtype="float32")[:, None, None]              # [L, 1, 1]
        k = ops.sum(self.c * b_bar * ops.exp(pos * dt_a), axis=-1)                # [L, H]
        # causal convolution: left-pad L-1 zeros, correlate with the time-reversed kernel
        u_pad = ops.pad(u, [[0, 0], [self.length - 1, 0], [0, 0]])
        return ops.depthwise_conv(u_pad, ops.flip(k, 0)[:, :, None], padding="valid") + u * self.d

    def get_config(self):
        return {**super().get_config(), "state_size": self.state_size}


def build_model(n_axes=6, window=50, channels=8, state_size=4, n_layers=2):
    inp = keras.Input(shape=(n_axes, window, 1))
    x = keras.layers.Permute((2, 1))(keras.layers.Reshape((n_axes, window))(inp))     # [50, 6]
    x = keras.layers.Conv1D(channels, 1)(x)
    for _ in range(n_layers):
        h = keras.layers.Activation("relu")(DiagonalSSM(state_size)(x))
        h = keras.layers.Conv1D(channels, 1, activation="relu")(h)
        x = keras.layers.Add()([x, h])
    x = keras.layers.GlobalAveragePooling1D()(x)
    return keras.Model(inp, keras.layers.Dense(2, activation="softmax")(x), name="TinySSM")
