"""Training orchestration: epoch loop, validation, mixed precision,
checkpointing, and early stopping."""

import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.config import Config
from training.losses import build_loss
from training.metrics import compute_binary_metrics
from training.optim import build_optimizer
from training.scheduler import build_scheduler
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.logger import ExperimentLogger


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Config,
        device: torch.device,
        resume_from: Optional[str] = None,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        self.criterion = build_loss(config.training)
        self.optimizer = build_optimizer(self.model, config.training)
        self.scheduler = build_scheduler(self.optimizer, config.training, len(train_loader))

        # Mixed precision only helps (and is only well-supported) on CUDA;
        # on CPU we simply run in full precision.
        self.amp_enabled = config.training.mixed_precision and device.type == "cuda"
        self.scaler = torch.amp.GradScaler(device=device.type, enabled=self.amp_enabled)

        experiment_dir = Path(config.paths.checkpoint_dir) / config.paths.experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        self.best_checkpoint_path = experiment_dir / "best.pt"
        self.last_checkpoint_path = experiment_dir / "last.pt"

        # append=True only when we are actually resuming a previous run of
        # THIS SAME experiment_name - see utils.logger for why this matters.
        self.logger = ExperimentLogger(
            config.paths.log_dir, config.paths.experiment_name, append=resume_from is not None
        )
        self.logger.log_config(config.to_dict())

        self.start_epoch = 0
        self.best_metric = -float("inf")
        self.epochs_without_improvement = 0
        self._backbone_unfrozen = False

        if resume_from is not None:
            self.resume(resume_from)

    def _unpack_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        clips, labels = batch
        clips = clips.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)
        return clips, labels

    def _maybe_unfreeze_backbone(self, epoch: int) -> None:
        """For `backbone_transformer`, unfreezes the last ResNet stages once
        the configured number of frozen warmup epochs has elapsed, then
        rebuilds the optimizer/scheduler so the newly-trainable parameters
        are included."""

        if self._backbone_unfrozen or not hasattr(self.model, "unfreeze_last_stages"):
            return
        if epoch < self.config.model.freeze_backbone_epochs:
            return

        self.model.unfreeze_last_stages(self.config.model.unfreeze_num_stages)
        self._backbone_unfrozen = True
        self.optimizer = build_optimizer(self.model, self.config.training)
        # Anneal over the epochs actually remaining, not the full run - a
        # fresh scheduler built against config.training.num_epochs here would
        # restart the whole warmup+cosine cycle from step 0, so the LR jumps
        # back up and never actually reaches its intended low value by the
        # real final epoch (visible as a repeating LR pattern in the logs).
        remaining_epochs = self.config.training.num_epochs - epoch
        self.scheduler = build_scheduler(
            self.optimizer, self.config.training, len(self.train_loader), num_epochs=remaining_epochs
        )
        print(f"[epoch {epoch + 1}] Unfroze backbone last stages for fine-tuning.")

    def train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        running_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1} [train]", leave=False)

        for batch in progress_bar:
            clips, labels = self._unpack_batch(batch)
            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                logits = self.model(clips)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            if self.config.training.grad_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            running_loss += loss.item() * clips.size(0)
            progress_bar.set_postfix(loss=loss.item(), lr=self.optimizer.param_groups[0]["lr"])

        return running_loss / len(self.train_loader.dataset)

    @torch.no_grad()
    def validate(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        running_loss = 0.0
        all_labels, all_probabilities = [], []
        progress_bar = tqdm(self.val_loader, desc=f"Epoch {epoch + 1} [val]", leave=False)

        for batch in progress_bar:
            clips, labels = self._unpack_batch(batch)
            with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                logits = self.model(clips)
                loss = self.criterion(logits, labels)

            running_loss += loss.item() * clips.size(0)
            all_labels.append(labels.cpu().numpy())
            all_probabilities.append(torch.sigmoid(logits).float().cpu().numpy())

        val_loss = running_loss / len(self.val_loader.dataset)
        y_true = np.concatenate(all_labels)
        y_prob = np.concatenate(all_probabilities)
        metrics = compute_binary_metrics(y_true, y_prob, threshold=self.config.training.decision_threshold)
        return val_loss, metrics

    def fit(self) -> None:
        num_epochs = self.config.training.num_epochs
        patience = self.config.training.early_stopping_patience
        monitor_metric = self.config.training.monitor_metric

        for epoch in range(self.start_epoch, num_epochs):
            self._maybe_unfreeze_backbone(epoch)

            epoch_start_time = time.time()
            train_loss = self.train_one_epoch(epoch)
            val_loss, val_metrics = self.validate(epoch)
            epoch_duration = time.time() - epoch_start_time

            log_payload = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "epoch_time_sec": epoch_duration,
                **{f"val_{name}": value for name, value in val_metrics.items()},
            }
            self.logger.log_epoch(epoch, log_payload)
            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_{monitor_metric}={val_metrics[monitor_metric]:.4f} "
                f"val_recall={val_metrics['recall']:.4f} val_auc={val_metrics['roc_auc']:.4f} "
                f"time={epoch_duration:.1f}s"
            )

            save_checkpoint(
                self.last_checkpoint_path,
                self.model,
                self.optimizer,
                self.scheduler,
                epoch,
                self.best_metric,
                self.config.to_dict(),
            )

            current_metric = val_metrics[monitor_metric]
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.epochs_without_improvement = 0
                save_checkpoint(
                    self.best_checkpoint_path,
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    self.best_metric,
                    self.config.to_dict(),
                )
            else:
                self.epochs_without_improvement += 1

            if self.epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs without improvement.")
                break

        self.logger.close()

    def resume(self, checkpoint_path: str) -> None:
        checkpoint = load_checkpoint(
            checkpoint_path, self.model, self.optimizer, self.scheduler, map_location=self.device
        )
        self.start_epoch = checkpoint["epoch"] + 1
        self.best_metric = checkpoint["best_metric"]
        print(f"Resumed from checkpoint at epoch {checkpoint['epoch']}, best_metric={self.best_metric:.4f}")