"""10-Layer CNN Architecture reproducing Table 8 and Table 6 experiments."""

from typing import Tuple
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_cnn_10_layers(input_dim: int = 30) -> keras.Sequential:
    """Build the 10-layer CNN sequential model matching Table 8 exactly.

    Parameters
    ----------
    input_dim : int, default=30
        Number of input features. When 30, yields exactly:
        Total params: 119,457
        Trainable params: 119,265
        Non-trainable params: 192

    Returns
    -------
    keras.Sequential
        Model matching Table 8.
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim, 1)),
        layers.Conv1D(filters=32, kernel_size=2, activation="relu", name="conv1d"),
        layers.BatchNormalization(name="batch_normalization"),
        layers.Dropout(0.2, name="dropout"),
        layers.Conv1D(filters=64, kernel_size=2, activation="relu", name="conv1d_1"),
        layers.BatchNormalization(name="batch_normalization_1"),
        layers.Dropout(0.5, name="dropout_1"),
        layers.Flatten(name="flatten"),
        layers.Dense(64, activation="relu", name="dense"),
        layers.Dropout(0.5, name="dropout_2"),
        layers.Dense(1, activation="sigmoid", name="dense_1")
    ], name="cnn_10_layers_model")

    return model


def verify_cnn_10_parameters(model: keras.Model) -> Tuple[int, int, int]:
    """Verify parameters match Table 8 (119,457 total, 119,265 trainable, 192 non-trainable)."""
    trainable_count = int(np.sum([tf.size(w).numpy() for w in model.trainable_weights]))
    non_trainable_count = int(np.sum([tf.size(w).numpy() for w in model.non_trainable_weights]))
    total_count = trainable_count + non_trainable_count
    return total_count, trainable_count, non_trainable_count
