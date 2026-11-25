"""
TinyTCN: 1x1 conv mixes the 6 axes -> residual blocks of dilated depthwise conv + 1x1 conv -> GAP -> softmax.
Causal padding, receptive field 1 + (kernel - 1) * sum(dilations) = 31 rows. 458 params.
"""
import keras


def build_model(n_axes=6, window=50, channels=8, kernel=3, dilations=(1, 2, 4, 8)):
    inp = keras.Input(shape=(n_axes, window, 1))
    x = keras.layers.Permute((2, 1))(keras.layers.Reshape((n_axes, window))(inp))     # [50, 6]
    x = keras.layers.Conv1D(channels, 1, activation="relu")(x)
    for d in dilations:
        h = keras.layers.ZeroPadding1D(((kernel - 1) * d, 0))(x)                        # causal
        h = keras.layers.DepthwiseConv1D(kernel, dilation_rate=d, use_bias=False)(h)
        h = keras.layers.Conv1D(channels, 1, activation="relu")(h)
        x = keras.layers.Add()([x, h])
    x = keras.layers.GlobalAveragePooling1D()(x)
    return keras.Model(inp, keras.layers.Dense(2, activation="softmax")(x), name="TinyTCN")
