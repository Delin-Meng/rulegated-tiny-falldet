"""
TinyEEGNet: EEGNet (Lawhern et al., 2018), the design TinyCNN names as its inspiration, scaled to 6 axes x 50 rows.
  temporal conv (1x16, F1 filters) -> BN -> depthwise "spatial" conv over the 6 axes (6x1, D per filter, valid)
  -> BN -> ReLU -> average pool (1x4) -> separable conv (1x8, F2 filters) -> BN -> ReLU -> average pool (1x4)
  -> GAP -> softmax
Differences from EEGNet: ReLU instead of ELU and no dropout (as in the other tiny models), GAP instead of flatten.
With F1=4, D=2, F2=8: 338 params (40 of them BatchNorm moving statistics, folded away in TFLite).
"""
import keras


def build_model(n_axes=6, window=50, f1=4, depth=2, f2=8):
    L = keras.layers
    inp = keras.Input(shape=(n_axes, window, 1))
    x = L.Conv2D(f1, (1, 16), padding="same", use_bias=False)(inp)               # temporal filters
    x = L.BatchNormalization()(x)
    x = L.DepthwiseConv2D((n_axes, 1), depth_multiplier=depth, use_bias=False)(x)  # spatial filters, 6 axes -> 1
    x = L.Activation("relu")(L.BatchNormalization()(x))
    x = L.AveragePooling2D((1, 4))(x)                                             # 50 -> 12 rows
    x = L.SeparableConv2D(f2, (1, 8), padding="same", use_bias=False)(x)          # depthwise 1x8 + pointwise 1x1
    x = L.Activation("relu")(L.BatchNormalization()(x))
    x = L.AveragePooling2D((1, 4))(x)                                             # 12 -> 3 rows
    x = L.GlobalAveragePooling2D()(x)
    return keras.Model(inp, L.Dense(2, activation="softmax")(x), name="TinyEEGNet")
