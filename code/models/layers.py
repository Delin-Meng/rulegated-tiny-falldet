"""
Input layers for baselines that do not use the pipeline's z-scored 6-axis input as is.
Both have no weights (nothing is added to the parameter count) and are saved with the model.
"""
import keras
import numpy as np


@keras.saving.register_keras_serializable()
class SelectAxes(keras.layers.Layer):
    """[batch, 6, rows, 1] -> [batch, axes, rows/step, 1]: the first `axes` input axes (AccX, AccY, AccZ for 3)
    and every `step`-th row (step 4: 100 Hz -> 25 Hz, plain decimation)."""

    def __init__(self, axes=3, step=1, **kwargs):
        super().__init__(**kwargs)
        self.axes, self.step = axes, step

    def call(self, x):
        return x[:, :self.axes, ::self.step, :]

    def compute_output_shape(self, shape):
        return (shape[0], self.axes, None if shape[2] is None else -(-shape[2] // self.step), shape[3])

    def get_config(self):
        return {**super().get_config(), "axes": self.axes, "step": self.step}


@keras.saving.register_keras_serializable()
class RawAccMagnitude(keras.layers.Layer):
    """[batch, 6, rows, 1] z-scored input -> [batch, rows, 1] acceleration magnitude |a| in g, undoing the
    z-score with the training statistics (mean, std per axis, the first three = accelerometer).
    Without statistics the input is taken as already raw."""

    def __init__(self, mean=None, std=None, **kwargs):
        super().__init__(**kwargs)
        self.mean = None if mean is None else [float(v) for v in np.ravel(mean)[:3]]
        self.std = None if std is None else [float(v) for v in np.ravel(std)[:3]]

    def call(self, x):
        acc = x[:, :3, :, 0]                                                    # [batch, 3, rows]
        if self.mean is not None:
            acc = acc * keras.ops.reshape(keras.ops.convert_to_tensor(self.std), (1, 3, 1)) + \
                keras.ops.reshape(keras.ops.convert_to_tensor(self.mean), (1, 3, 1))
        return keras.ops.sqrt(keras.ops.sum(acc ** 2, axis=1))[..., None]     # [batch, rows, 1]

    def compute_output_shape(self, shape):
        return (shape[0], shape[2], 1)

    def get_config(self):
        return {**super().get_config(), "mean": self.mean, "std": self.std}
