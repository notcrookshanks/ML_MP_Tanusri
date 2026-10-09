"""Training orchestrator for Deep Learning and CNN models."""

import time
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.evaluation.metrics import calculate_all_metrics
from src.models.model_factory import create_dl_model
from src.training.callbacks import EpochMetricsTracker


def train_and_evaluate_dl_model(
    architecture: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 20,
    batch_size: int = 2048,
    learning_rate: float = 0.001,
    random_state: int = 42,
    dataset_variant: str = "imbalanced",
    use_careful_bias: bool = False,
    verbose: int = 1
) -> Dict[str, Any]:
    """Train a deep learning model (Baseline MLP or CNN) and evaluate performance.

    Parameters
    ----------
    architecture : str
        Model architecture: 'baseline_nn', 'cnn_10_layers', 'cnn_14_layers',
        'cnn_17_layers', 'cnn_20_layers'.
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.
    X_test : np.ndarray
        Validation/testing features.
    y_test : np.ndarray
        Validation/testing labels.
    epochs : int, default=20
        Number of training epochs.
    batch_size : int, default=2048
        Batch size (paper recommends >2048 for imbalanced credit card data).
    learning_rate : float, default=0.001
        Adam optimizer learning rate.
    random_state : int, default=42
        Reproducibility seed.
    dataset_variant : str
        Label for the dataset variant.
    use_careful_bias : bool, default=False
        Whether to initialize output bias to log(pos/neg).
    verbose : int, default=1
        Keras training verbosity.

    Returns
    -------
    dict
        Evaluation results, training history, predictions, and runtime.
    """
    tf.keras.utils.set_random_seed(random_state)

    is_cnn = "cnn" in architecture.lower()
    input_dim = X_train.shape[1]

    # Prepare inputs: 3D (samples, features, 1) for Conv1D, 2D for MLP
    if is_cnn:
        if X_train.ndim == 2:
            X_tr = np.expand_dims(X_train, axis=-1)
            X_te = np.expand_dims(X_test, axis=-1)
        else:
            X_tr = X_train
            X_te = X_test
    else:
        if X_train.ndim == 3:
            X_tr = np.squeeze(X_train, axis=-1)
            X_te = np.squeeze(X_test, axis=-1)
        else:
            X_tr = X_train
            X_te = X_test

    output_bias = None
    if use_careful_bias:
        pos = int(np.sum(y_train == 1))
        neg = int(np.sum(y_train == 0))
        output_bias = float(np.log(max(pos, 1) / max(neg, 1)))

    model = create_dl_model(
        architecture=architecture,
        input_dim=input_dim,
        compile_model=True,
        learning_rate=learning_rate,
        output_bias=output_bias
    )

    tracker = EpochMetricsTracker()

    start_time = time.time()
    history = model.fit(
        X_tr,
        y_train,
        validation_data=(X_te, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[tracker],
        verbose=verbose
    )
    total_train_time = time.time() - start_time

    # Inference
    infer_start = time.time()
    y_prob = model.predict(X_te, batch_size=batch_size, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    infer_time = time.time() - infer_start

    metrics = calculate_all_metrics(y_true=y_test, y_pred=y_pred, y_prob=y_prob)

    # Final epoch training accuracy and validation accuracy
    final_train_acc = float(history.history.get("accuracy", [0.0])[-1])
    final_val_acc = float(history.history.get("val_accuracy", [0.0])[-1])

    result: Dict[str, Any] = {
        "model": architecture,
        "dataset_variant": dataset_variant,
        "epochs": epochs,
        "batch_size": batch_size,
        "random_state": random_state,
        "train_time_sec": round(total_train_time, 3),
        "infer_time_sec": round(infer_time, 3),
        "total_runtime_sec": round(total_train_time + infer_time, 3),
        "final_train_accuracy_pct": round(final_train_acc * 100, 4),
        "final_val_accuracy_pct": round(final_val_acc * 100, 4),
        "status": "COMPLETED",
        "history": tracker.history,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "model_instance": model,
        **metrics
    }

    return result
