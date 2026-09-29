"""Learning-rate schedule: linear warmup followed by cosine decay.

Stepped once per training batch (not per epoch) so the warmup phase is
measured in optimisation steps regardless of dataset size.
"""

import math
from typing import Optional

import torch


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    training_cfg,
    steps_per_epoch: int,
    num_epochs: Optional[int] = None,
) -> torch.optim.lr_scheduler.LambdaLR:
    """`num_epochs` defaults to `training_cfg.num_epochs`, but callers that
    rebuild the scheduler partway through training (e.g. `Trainer` unfreezing
    the backbone) should pass the number of epochs *remaining*, so the cosine
    decay still finishes at the real final epoch instead of restarting a full
    schedule from wherever training happens to be."""

    num_epochs = training_cfg.num_epochs if num_epochs is None else num_epochs
    total_steps = max(1, steps_per_epoch * num_epochs)
    warmup_steps = int(total_steps * training_cfg.warmup_ratio)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)