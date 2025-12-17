"""
Reproducibility helpers.
"""
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def set_global_seed(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Determinism is more relevant once we use torch/tf.
    # Notebook 00 keeps this placeholder for consistency across notebooks.
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
