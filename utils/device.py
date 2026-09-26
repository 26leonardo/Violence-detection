"""Device management helpers."""

import torch


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Returns the CUDA device if available and requested, otherwise CPU."""

    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
