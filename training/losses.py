"""Binary classification losses."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    """Down-weighs easy (already well-classified) examples so that hard
    examples dominate the gradient - useful when violent/non-violent classes
    are imbalanced (e.g. when generalising to an untrimmed dataset)."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = probabilities * targets + (1 - probabilities) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - p_t) ** self.gamma * ce_loss
        return loss.mean()


def build_loss(training_cfg) -> nn.Module:
    if training_cfg.loss == "bce":
        return nn.BCEWithLogitsLoss()
    if training_cfg.loss == "focal":
        return BinaryFocalLoss(gamma=training_cfg.focal_gamma, alpha=training_cfg.focal_alpha)
    raise ValueError(f"Unknown loss type: {training_cfg.loss!r}")
