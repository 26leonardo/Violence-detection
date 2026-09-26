"""Checkpoint saving and loading.

Checkpoints bundle the model, optimizer and scheduler state together with the
epoch, best metric value, and the full experiment configuration, so a run can
be resumed or audited without any external bookkeeping.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def save_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[LRScheduler],
    epoch: int,
    best_metric: float,
    config_dict: Dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "best_metric": best_metric,
            "config": config_dict,
        },
        path,
    )


def load_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[LRScheduler] = None,
    map_location: Union[str, torch.device] = "cpu",
) -> Dict[str, Any]:
    # weights_only=False is required because this checkpoint stores optimizer
    # state and a plain config dict alongside the tensors; only load
    # checkpoints you trust (e.g. ones your own training run produced).
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    return checkpoint
