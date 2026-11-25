"""
TinyTransformer: attention baseline.
  patch embedding: Conv1D over the 6 axes with kernel = stride = 5 -> 10 tokens of width 8, + learned positions
  one pre-norm Transformer block: x + MHA(LN(x)) (2 heads, key dim 4), then x + FFN(LN(x)) (8 -> 16 -> 8)
  -> average over tokens -> softmax. 946 params.
"""
import keras


@keras.saving.register_keras_serializable()
class PositionEmbedding(keras.layers.Layer):
    """Adds a learned vector per token position."""

    def build(self, input_shape):
        self.pos = self.add_weight(shape=(input_shape[1], input_shape[2]),
                                   initializer=keras.initializers.RandomNormal(stddev=0.02))

    def call(self, x):
        return x + self.pos


def build_model(n_axes=6, window=50, width=8, patch=5, heads=2, key_dim=4, ffn=16):
    L = keras.layers
    inp = keras.Input(shape=(n_axes, window, 1))
    x = L.Permute((2, 1))(L.Reshape((n_axes, window))(inp))                      # [50, 6]
    x = PositionEmbedding()(L.Conv1D(width, patch, strides=patch)(x))           # [10, 8]
    h = L.LayerNormalization()(x)
    x = L.Add()([x, L.MultiHeadAttention(num_heads=heads, key_dim=key_dim)(h, h)])
    h = L.LayerNormalization()(x)
    h = L.Dense(width)(L.Dense(ffn, activation="relu")(h))
    x = L.GlobalAveragePooling1D()(L.Add()([x, h]))
    return keras.Model(inp, L.Dense(2, activation="softmax")(x), name="TinyTransformer")
