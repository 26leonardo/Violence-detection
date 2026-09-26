"""Optimizer construction.

`backbone_transformer` uses AdamW with two parameter groups (a low learning
rate for any unfrozen backbone layers, a normal one for the new head) since
weight decay in AdamW is decoupled from the gradient moments and applied
directly to the weights - unlike plain SGD, where L2 regularisation and
weight decay coincide. `lightweight_tsm` uses SGD with Nesterov momentum,
which tends to outperform Adam on CNN image/video classification when
properly tuned.
"""

import torch
import torch.nn as nn


def build_optimizer(model: nn.Module, training_cfg) -> torch.optim.Optimizer:
    if training_cfg.optimizer == "adamw":
        backbone_params = [
            p for n, p in model.named_parameters() if n.startswith("backbone") and p.requires_grad
        ]
        other_params = [
            p for n, p in model.named_parameters() if not n.startswith("backbone") and p.requires_grad
        ]
        param_groups = [{"params": other_params, "lr": training_cfg.learning_rate}]
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": training_cfg.backbone_learning_rate})
        return torch.optim.AdamW(param_groups, weight_decay=training_cfg.weight_decay, betas=(0.9, 0.999))

    if training_cfg.optimizer == "sgd":
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.SGD(
            trainable_params,
            lr=training_cfg.learning_rate,
            momentum=0.9,
            nesterov=True,
            weight_decay=training_cfg.weight_decay,
        )

    raise ValueError(f"Unknown optimizer: {training_cfg.optimizer!r}")
