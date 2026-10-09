"""Training callbacks and history tracking for deep learning models."""

import time
from typing import Dict, List
import tensorflow as tf
from tensorflow import keras


class EpochMetricsTracker(keras.callbacks.Callback):
    """Callback to capture detailed epoch-by-epoch history including durations."""

    def __init__(self):
        super().__init__()
        self.history: Dict[str, List[float]] = {
            "epoch": [],
            "loss": [],
            "val_loss": [],
            "accuracy": [],
            "val_accuracy": [],
            "precision": [],
            "val_precision": [],
            "recall": [],
            "val_recall": [],
            "epoch_duration_sec": []
        }
        self._epoch_start_time = 0.0

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        duration = time.time() - self._epoch_start_time
        logs = logs or {}

        self.history["epoch"].append(epoch + 1)
        self.history["epoch_duration_sec"].append(round(duration, 3))
        self.history["loss"].append(float(logs.get("loss", 0.0)))
        self.history["val_loss"].append(float(logs.get("val_loss", 0.0)))
        self.history["accuracy"].append(float(logs.get("accuracy", 0.0)))
        self.history["val_accuracy"].append(float(logs.get("val_accuracy", 0.0)))
        self.history["precision"].append(float(logs.get("precision", 0.0)))
        self.history["val_precision"].append(float(logs.get("val_precision", 0.0)))
        self.history["recall"].append(float(logs.get("recall", 0.0)))
        self.history["val_recall"].append(float(logs.get("val_recall", 0.0)))
