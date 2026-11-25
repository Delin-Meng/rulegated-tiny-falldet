"""
Musci et al., "Online fall detection using recurrent neural networks on smart wearable devices",
IEEE Trans. Emerg. Top. Comput. 9 (2021) 1276-1289 (TinyCNN Table 5: LSTM, SisFall, 2,322 params, STM32).
No official code; reproduced from the paper (Fig. 1, Sec. III-B, V-A, Table 2) and its implementation paper
(Torti et al., DSD 2018, Algorithm 1). Final model chosen in Sec. V-A: k = 1 LSTM layer, n = 16 units,
accelerometer + gyroscope (6 channels):
  per-time-step Dense(16, ReLU) -> BatchNorm -> Dropout -> LSTM(16), last step -> Dropout -> Dense(2) softmax
6*16+16 = 112 + BN 64 (32 moving statistics) + 4*16*(16+16+1) = 2,112 + 16*2+2 = 34 -> 2,322 params, the figure
in TinyCNN's Table 5 (2 classes, BatchNorm statistics counted).
As in the paper: L2 1.5e-3 on all weights (TensorFlow l2_loss convention -> Keras l2(7.5e-4)), dropout keep
probability 0.8 (Keras rate 0.2).
Adapted to this pipeline: 0.5 s windows at 100 Hz instead of 1.28 s (256 samples) at 200 Hz; 2 classes instead
of BKG / ALERT / FALL; the paper's low-pass Butterworth filter is not applied (order and cutoff not given);
common training protocol (Adam 5e-4, focal loss, 200 epochs) instead of Adam 2.5e-3, weighted cross-entropy,
300 epochs. LSTM unrolled for TFLite.
"""
import keras

L2 = keras.regularizers.l2(7.5e-4)


def build_model(n_axes=6, window=50, units=16, dropout=0.2):
    L = keras.layers
    inp = keras.Input(shape=(n_axes, window, 1))
    x = L.Permute((2, 1))(L.Reshape((n_axes, window))(inp))                      # [50, 6]
    x = L.Dense(units, activation="relu", kernel_regularizer=L2, bias_regularizer=L2)(x)   # FC1 per step
    x = L.Dropout(dropout)(L.BatchNormalization()(x))
    x = L.LSTM(units, unroll=True, kernel_regularizer=L2, recurrent_regularizer=L2, bias_regularizer=L2)(x)
    x = L.Dropout(dropout)(x)
    out = L.Dense(2, activation="softmax", kernel_regularizer=L2, bias_regularizer=L2)(x)
    return keras.Model(inp, out, name="Musci2021")
