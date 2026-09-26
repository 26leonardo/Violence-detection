"""Learning-rate schedule: linear warmup followed by cosine decay.

Stepped once per training batch (not per epoch) so the warmup phase is
measured in optimisation steps regardless of dataset size.
"""

import math

import torch


def build_scheduler(
    optimizer: torch.optim.Optimizer, training_cfg, steps_per_epoch: int
) -> torch.optim.lr_scheduler.LambdaLR:
    total_steps = max(1, steps_per_epoch * training_cfg.num_epochs)
    warmup_steps = int(total_steps * training_cfg.warmup_ratio)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
