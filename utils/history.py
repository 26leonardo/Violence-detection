"""Loads and plots the JSONL training history written by `utils.logger.ExperimentLogger`.

Kept separate from that module because this is notebook/analysis-facing
code, not something the training loop itself depends on.
"""

import json
from pathlib import Path
from typing import Dict, List, Union

import matplotlib.pyplot as plt


def load_history(log_dir: Union[str, Path], experiment_name: str) -> List[Dict[str, float]]:
    """Reads history.jsonl for one experiment, returning one dict per epoch.

    Defensive against history files written before `Trainer` started
    truncating stale logs on a fresh (non-resumed) run: if two separate
    training runs were ever logged under the same experiment name, their
    records sit back-to-back with the epoch counter jumping back down to 0
    partway through the file. Plotting that directly draws a line that
    zig-zags backwards in time. Here, if such a reset is detected, only the
    most recent contiguous run (the one after the last reset) is returned.
    """
    history_path = Path(log_dir) / experiment_name / "history.jsonl"
    if not history_path.exists():
        raise FileNotFoundError(f"No history found at {history_path}")

    with open(history_path) as history_file:
        records = [json.loads(line) for line in history_file if line.strip()]
    if not records:
        raise ValueError(f"{history_path} exists but contains no records.")

    last_run_start = 0
    for i in range(1, len(records)):
        if records[i]["epoch"] <= records[i - 1]["epoch"]:
            last_run_start = i
    return records[last_run_start:]


def plot_training_curves(records: List[Dict[str, float]], title: str = "") -> None:
    """Plots loss curves and validation metric curves side by side."""

    epochs = [r["epoch"] for r in records]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, [r["train_loss"] for r in records], label="train loss")
    axes[0].plot(epochs, [r["val_loss"] for r in records], label="val loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title(f"{title} - loss".strip(" -"))
    axes[0].legend()

    for metric_key, label in [("val_recall", "val recall"), ("val_f1", "val F1"), ("val_roc_auc", "val ROC-AUC")]:
        if metric_key in records[0]:
            axes[1].plot(epochs, [r[metric_key] for r in records], label=label)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("score")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title(f"{title} - validation metrics".strip(" -"))
    axes[1].legend()

    fig.tight_layout()
    plt.show()


def plot_comparison(
    histories: Dict[str, List[Dict[str, float]]], metric_key: str = "val_f1"
) -> None:
    """Overlays one metric across several experiments, e.g. to compare both
    architectures on the same axes: `plot_comparison({"lightweight_tsm": run_a,
    "backbone_transformer": run_b}, metric_key="val_f1")`."""

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, records in histories.items():
        epochs = [r["epoch"] for r in records]
        ax.plot(epochs, [r[metric_key] for r in records], label=name)
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric_key)
    ax.set_title(f"{metric_key} across experiments")
    ax.legend()
    plt.show()
