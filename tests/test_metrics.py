"""Unit tests verifying evaluation metric calculations against known analytical examples."""

import numpy as np
import pytest

from src.evaluation.metrics import calculate_all_metrics


def test_metrics_perfect_predictions():
    """Verify metrics for 100% correct predictions."""
    y_true = np.array([0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.05, 0.9, 0.95])

    metrics = calculate_all_metrics(y_true, y_pred, y_prob)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0
    assert metrics["false_positive_rate"] == 0.0
    assert metrics["false_negative_rate"] == 0.0


def test_metrics_known_imbalanced_counts():
    """Verify metrics on a known confusion matrix setup."""
    # Setup: 100 negatives, 10 positives
    # TN=95, FP=5, FN=2, TP=8
    y_true = np.array([0] * 100 + [1] * 10)
    y_pred = np.array([0] * 95 + [1] * 5 + [0] * 2 + [1] * 8)

    metrics = calculate_all_metrics(y_true, y_pred)

    assert metrics["true_negatives"] == 95
    assert metrics["false_positives"] == 5
    assert metrics["false_negatives"] == 2
    assert metrics["true_positives"] == 8

    # Precision = TP / (TP + FP) = 8 / (8 + 5) = 8/13 = 0.6154
    assert round(metrics["precision"], 4) == round(8 / 13, 4)
    # Recall = TP / (TP + FN) = 8 / (8 + 2) = 0.80
    assert round(metrics["recall"], 4) == 0.8000
    # FPR = FP / (TN + FP) = 5 / 100 = 0.05
    assert round(metrics["false_positive_rate"], 4) == 0.0500
    # FNR = FN / (TP + FN) = 2 / 10 = 0.20
    assert round(metrics["false_negative_rate"], 4) == 0.2000


def test_metrics_zero_division_safety():
    """Ensure no crash when model predicts all zeros."""
    y_true = np.array([0, 0, 1])
    y_pred = np.array([0, 0, 0])

    metrics = calculate_all_metrics(y_true, y_pred)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1_score"] == 0.0
