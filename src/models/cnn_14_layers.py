"""Proposed 14-Layer CNN Architecture reproducing Section IV-C-11-a."""

from typing import Tuple
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_cnn_14_layers(input_dim: int = 30) -> keras.Sequential:
    """Build the proposed 14-layer CNN sequential model.

    Architecture (14 Layers):
    1. Conv1D (32 filters, kernel size 2, ReLU)
    2. Batch Normalization
    3. Dropout (0.2)
    4. Conv1D (64 filters, kernel size 2, ReLU)
    5. Batch Normalization
    6. Dropout (0.5)
    7. Flatten
    8. Dense (100 units, ReLU)
    9. Dropout (0.5)
    10. Dense (50 units, ReLU)
    11. Dropout (0.25)
    12. Dense (25 units, ReLU)
    13. Dropout (0.25)
    14. Dense (1 unit, Sigmoid)

    Parameters
    ----------
    input_dim : int, default=30
        Number of input predictor features.

    Returns
    -------
    keras.Sequential
        14-layer CNN model.
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim, 1)),
        layers.Conv1D(filters=32, kernel_size=2, activation="relu", name="conv1d_1"),
        layers.BatchNormalization(name="bn_1"),
        layers.Dropout(0.2, name="dropout_1"),
        layers.Conv1D(filters=64, kernel_size=2, activation="relu", name="conv1d_2"),
        layers.BatchNormalization(name="bn_2"),
        layers.Dropout(0.5, name="dropout_2"),
        layers.Flatten(name="flatten"),
        layers.Dense(100, activation="relu", name="dense_100"),
        layers.Dropout(0.5, name="dropout_3"),
        layers.Dense(50, activation="relu", name="dense_50"),
        layers.Dropout(0.25, name="dropout_4"),
        layers.Dense(25, activation="relu", name="dense_25"),
        layers.Dropout(0.25, name="dropout_5"),
        layers.Dense(1, activation="sigmoid", name="dense_output")
    ], name="cnn_14_layers_model")

    return model
