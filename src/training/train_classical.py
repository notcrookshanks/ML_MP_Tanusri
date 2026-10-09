"""Training orchestrator for traditional Machine Learning classifiers."""

import time
from typing import Any, Dict, Optional
import numpy as np
from sklearn.base import BaseEstimator

from src.evaluation.metrics import calculate_all_metrics
from src.models.model_factory import create_classical_model


def train_and_evaluate_classical_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
    dataset_variant: str = "standardized_amount",
    svm_max_samples: Optional[int] = 30000,
    **kwargs: Any
) -> Dict[str, Any]:
    """Train a classical ML model and compute evaluation metrics.

    Parameters
    ----------
    model_name : str
        Name of model: 'decision_tree', 'knn', 'logistic_regression',
        'random_forest', 'svm', 'xgboost', 'elm'.
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training labels.
    X_test : np.ndarray
        Testing feature matrix.
    y_test : np.ndarray
        Testing labels.
    random_state : int, default=42
        Reproducibility seed.
    dataset_variant : str
        Label for the dataset variant (e.g. 'standardized_amount').
    svm_max_samples : int, optional
        Subsample limit for SVM training if dataset is large, to prevent timeout.

    Returns
    -------
    dict
        Comprehensive results dictionary including metrics, predictions, and runtime.
    """
    model = create_classical_model(model_name, random_state=random_state, **kwargs)

    # Handle large sample size for quadratic SVM if specified
    X_fit = X_train
    y_fit = y_train
    if model_name.lower() == "svm" and svm_max_samples is not None and len(X_train) > svm_max_samples:
        rng = np.random.RandomState(random_state)
        # Ensure all fraud samples are included in the subsample
        fraud_idx = np.where(y_train == 1)[0]
        legit_idx = np.where(y_train == 0)[0]
        sample_legit = rng.choice(legit_idx, size=min(svm_max_samples - len(fraud_idx), len(legit_idx)), replace=False)
        sub_idx = np.concatenate([fraud_idx, sample_legit])
        rng.shuffle(sub_idx)
        X_fit = X_train[sub_idx]
        y_fit = y_train[sub_idx]

    start_time = time.time()
    model.fit(X_fit, y_fit)
    train_time = time.time() - start_time

    # Predict
    infer_start = time.time()
    y_pred = model.predict(X_test)
    infer_time = time.time() - infer_start

    # Predict Probabilities
    y_prob = None
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X_test)
            if probs.ndim == 2 and probs.shape[1] >= 2:
                y_prob = probs[:, 1]
            else:
                y_prob = probs.ravel()
        except Exception:
            y_prob = None
    elif hasattr(model, "decision_function"):
        try:
            df = model.decision_function(X_test)
            y_prob = 1.0 / (1.0 + np.exp(-df))
        except Exception:
            y_prob = None

    metrics = calculate_all_metrics(y_true=y_test, y_pred=y_pred, y_prob=y_prob)

    result: Dict[str, Any] = {
        "model": model_name,
        "dataset_variant": dataset_variant,
        "random_state": random_state,
        "train_time_sec": round(train_time, 3),
        "infer_time_sec": round(infer_time, 3),
        "total_runtime_sec": round(train_time + infer_time, 3),
        "status": "COMPLETED",
        "y_pred": y_pred,
        "y_prob": y_prob,
        **metrics
    }

    return result
