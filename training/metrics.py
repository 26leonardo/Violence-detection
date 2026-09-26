"""Metrics for the violent / non-violent binary classification task.

Accuracy, precision, recall, and F1 summarise threshold-based performance
and are the standard choice for a balanced (or near-balanced) binary
classification task such as RWF-2000. ROC-AUC is threshold-independent and
useful for comparing models or for tuning the decision threshold later.
Regression metrics (MAE, MSE, RMSE, R^2) are intentionally not computed:
the target is a binary label, not a continuous quantity, so they would not
be meaningful here.
"""

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
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
        "roc_auc": float(auc),
    }


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)
