"""Comprehensive evaluation metrics for imbalanced credit card fraud detection."""

from typing import Any, Dict, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)


def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """Calculate all performance evaluation metrics matching paper definitions.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels (0 or 1).
    y_pred : np.ndarray
        Predicted binary labels (0 or 1).
    y_prob : np.ndarray, optional
        Predicted probabilities for positive class (Class=1).
    threshold : float, default=0.5
        Decision threshold for binary classification.

    Returns
    -------
    dict
        Dictionary containing all metrics and confusion counts.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    if y_prob is not None:
        y_prob = np.asarray(y_prob).ravel()

    # Confusion Matrix: tn, fp, fn, tp
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    total_samples = len(y_true)
    total_positives = int(tp + fn)
    total_negatives = int(tn + fp)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # Rates
    fpr = float(fp / max(total_negatives, 1))
    fnr = float(fn / max(total_positives, 1))

    # Area metrics
    auc_val: Optional[float] = None
    prc_val: Optional[float] = None

    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            auc_val = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            auc_val = None
        try:
            prc_val = float(average_precision_score(y_true, y_prob))
        except ValueError:
            prc_val = None

    return {
        "accuracy": acc,
        "accuracy_pct": round(acc * 100, 4),
        "precision": prec,
        "precision_pct": round(prec * 100, 4),
        "recall": rec,
        "recall_pct": round(rec * 100, 4),
        "f1_score": f1,
        "f1_score_pct": round(f1 * 100, 4),
        "roc_auc": auc_val,
        "average_precision_prc": prc_val,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "total_test_samples": total_samples,
        "total_fraud_samples": total_positives,
        "threshold": threshold
    }
