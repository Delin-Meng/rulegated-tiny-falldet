"""
Model registry. Each entry: name -> (module in this package, build_model arguments, display name).
To try a new variant, add one line here and pass its name to `--models`.
"""
import importlib
import inspect

VARIANTS = {
    "tinycnn":       ("tinycnn", {}, "TinyCNN series"),
    "tinycnn_valid": ("tinycnn", {"spatial_padding": "valid"}, "TinyCNN series (valid)"),
    "tinytcn":       ("tinytcn", {}, "TinyTCN"),
    "tinyssm":       ("tinyssm", {}, "TinySSM"),
    "cnnlstm":       ("cnnlstm", {}, "CNN-LSTM"),          # recurrent baseline, 1,426 params
    "tinyeegnet":    ("tinyeegnet", {}, "TinyEEGNet"),     # the design TinyCNN cites as its inspiration
    "tinytransformer": ("tinytransformer", {}, "TinyTransformer"),   # attention baseline
    # baselines of TinyCNN's Table 5, reproduced (see each file for the source and the adaptations)
    "musci2021":        ("musci2021", {}, "Musci 2021 (LSTM)"),              # 2,322 params, STM32
    # not in the experiments for now; files in "New folder" (move them back up to use them again):
    # "lunaperejon2019":  ("lunaperejon2019", {}, "Luna-Perejon 2019 (LSTM)"),  # official code, STM32
    # "kraft2020":        ("kraft2020", {}, "Kraft 2020 (CNN)"),                # |a| input, Nordic MCU
    # "santoyoramon2022": ("santoyoramon2022", {}, "Santoyo-Ramon 2022 (CNN)"), # 134,338 params
    # TinyTCN size sweep (P1 item 5): channels (default 8) and number of dilated blocks (default 4: 1, 2, 4, 8)
    "tinytcn_c4":    ("tinytcn", {"channels": 4}, "TinyTCN (4 ch)"),             # 166 params
    "tinytcn_c6":    ("tinytcn", {"channels": 6}, "TinyTCN (6 ch)"),             # 296 params
    "tinytcn_c12":   ("tinytcn", {"channels": 12}, "TinyTCN (12 ch)"),           # 878 params
    "tinytcn_c16":   ("tinytcn", {"channels": 16}, "TinyTCN (16 ch)"),           # 1,426 params
    "tinytcn_b2":    ("tinytcn", {"dilations": (1, 2)}, "TinyTCN (2 blocks)"),   # 266 params, receptive field 7
    "tinytcn_b3":    ("tinytcn", {"dilations": (1, 2, 4)}, "TinyTCN (3 blocks)"),  # 362 params, receptive field 15
}
DEFAULT = ["tinycnn", "tinytcn", "tinyssm"]
MAIN = ["tinycnn", "tinytcn", "tinyssm", "cnnlstm", "tinyeegnet", "tinytransformer", "musci2021"]   # the 7 models
SWEEP_TINYTCN = ["tinytcn_c4", "tinytcn_c6", "tinytcn", "tinytcn_c12", "tinytcn_c16", "tinytcn_b2", "tinytcn_b3"]


def build(name, n_axes, window, norm=None):
    """norm: (mean, std) of the training data's z-score, passed to models whose build_model takes `norm`
    (baselines that need raw values, e.g. the acceleration magnitude); ignored by the others."""
    module, kwargs, _ = VARIANTS[name]
    fn = importlib.import_module(f"models.{module}").build_model
    if norm is not None and "norm" in inspect.signature(fn).parameters:
        kwargs = {**kwargs, "norm": norm}
    return fn(n_axes=n_axes, window=window, **kwargs)


def display_name(name):
    return VARIANTS[name][2] if name in VARIANTS else name


def register_custom_layers():
    """Import modules that define custom layers, needed before loading saved .keras models."""
    for module in ("tinyssm", "tinytransformer", "layers"):
        importlib.import_module(f"models.{module}")
