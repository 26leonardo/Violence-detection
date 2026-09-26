"""Small model-inspection helpers."""

from typing import Tuple

import torch.nn as nn


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Returns (total_parameters, trainable_parameters)."""

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable
