"""Metrics for the violent / non-violent binary classification task.

Accuracy, precision, recall, and F1 summarise threshold-based performance
and are the standard choice for a balanced (or near-balanced) binary
classification task such as RWF-2000. F2 is included alongside F1 because,
for violence detection specifically, a false negative (a real fight
labelled "safe") is a worse mistake than a false positive (a false alarm on
a calm clip) - F2 weighs recall twice as heavily as precision, so a model
that misses fewer real fights scores higher even if it raises more false
alarms. ROC-AUC is threshold-independent and useful for comparing models or
for tuning the decision threshold later. Regression metrics (MAE, MSE,
RMSE, R^2) are intentionally not computed: the target is a binary label,
not a continuous quantity, so they would not be meaningful here.
"""

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    fbeta_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    f2 = fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        # ROC-AUC is undefined when a batch/split contains only one class.
        auc = float("nan")

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f2": float(f2),
        "roc_auc": float(auc),
    }


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)


def find_threshold_for_recall(y_true: np.ndarray, y_prob: np.ndarray, target_recall: float = 0.9) -> float:
    """Returns the highest decision threshold that still achieves at least
    `target_recall` on this data.

    The default threshold of 0.5 is an arbitrary midpoint with no special
    status - for a task where missing a violent clip is costly, it's
    reasonable to deliberately lower it. This scans the precision-recall
    curve for the least aggressive threshold (the one that still flags the
    fewest calm clips as violent) that reaches the recall you require, so
    you're not lowering it further than necessary.
    """
    if not 0.0 < target_recall <= 1.0:
        raise ValueError(f"target_recall must be in (0, 1], got {target_recall}")

    _, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns one more precision/recall point than
    # threshold (it appends the (precision=1, recall=0) endpoint), so align
    # by dropping that last point before comparing against `thresholds`.
    reachable = recalls[:-1] >= target_recall
    if not reachable.any():
        return 0.0  # even flagging every clip as violent can't reach target_recall
    return float(thresholds[reachable].max())
