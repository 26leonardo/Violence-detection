"""Reproducibility helpers.

Note on unavoidable non-determinism: even with every seed fixed and cuDNN set
to deterministic mode, a handful of CUDA operations (e.g. some scatter/index
ops used internally by adaptive pooling and certain interpolation modes) do
not have a deterministic backward implementation. For this project's
operations (standard convolutions, batch norm, linear layers, multi-head
attention) results are deterministic across runs on the same hardware and
software versions.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seeds Python, NumPy, and PyTorch (CPU and all CUDA devices)."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
