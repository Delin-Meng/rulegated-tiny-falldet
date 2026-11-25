"""
TinyCNN (Yu et al., 2023, BSPC 86:105325).
Temporal conv (1x16, 8) -> spatial conv over 6 axes (6x1, 8) -> GAP -> softmax. 546 params.
"""
import keras


def build_model(n_axes=6, window=50, spatial_padding="same"):
    """spatial_padding="same" keeps the 6 x 50 maps as the paper states (Sec. 2.2, Fig. 2; 153,616 MACs);
    "valid" collapses the 6 axes to 1 row as in EEGNet (57,616 MACs). Both have 546 parameters."""
    return keras.Sequential([
        keras.Input(shape=(n_axes, window, 1)),
        keras.layers.Conv2D(8, (1, 16), padding="same", activation="relu"),
        keras.layers.Conv2D(8, (n_axes, 1), padding=spatial_padding, activation="relu"),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(2, activation="softmax"),
    ], name="TinyCNN")
