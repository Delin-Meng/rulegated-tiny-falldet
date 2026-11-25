"""
CNN-LSTM baseline (recurrent comparison): Conv1D (8 filters, kernel 8) over the 6 axes -> LSTM(12), last state
-> softmax. 6*8*8+8 = 392 + 4*12*(8+12+1) = 1,008 + 12*2+2 = 26 -> 1,426 params.
"""
import keras


def build_model(n_axes=6, window=50, filters=8, kernel=8, units=12):
    inp = keras.Input(shape=(n_axes, window, 1))
    x = keras.layers.Permute((2, 1))(keras.layers.Reshape((n_axes, window))(inp))     # [50, 6]
    x = keras.layers.Conv1D(filters, kernel, padding="same", activation="relu")(x)
    # unrolled over the fixed 50 steps: same function, but TFLite gets plain FULLY_CONNECTED / element-wise ops
    # instead of a WHILE loop (which hides the MACs and crashes full-INT8 calibration)
    x = keras.layers.LSTM(units, unroll=True)(x)
    return keras.Model(inp, keras.layers.Dense(2, activation="softmax")(x), name="CNN-LSTM")
