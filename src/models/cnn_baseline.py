"""Baseline Neural Network implementation reproducing Table 9 and Section IV-C-4."""

from typing import Optional, Tuple
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_baseline_cnn(
    input_dim: int = 29,
    hidden_units: int = 16,
    dropout_rate: float = 0.5,
    output_bias: Optional[float] = None
) -> keras.Sequential:
    """Build the baseline neural network architecture matching Table 9 (497 parameters).

    Parameters
    ----------
    input_dim : int, default=29
        Input feature dimension. When 29 (V1-V28 + Amount), total parameters = 497.
        When 30 (Time + V1-V28 + Amount), total parameters = 513.
    hidden_units : int, default=16
        Number of units in the first Dense layer.
    dropout_rate : float, default=0.5
        Dropout probability.
    output_bias : float, optional
        Careful bias initialization value, typically log(pos / neg).

    Returns
    -------
    keras.Sequential
        Compiled or uncompiled Keras sequential model.
    """
    bias_init = (
        keras.initializers.Constant(output_bias)
        if output_bias is not None
        else "zeros"
    )

    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(hidden_units, activation="relu", name="dense"),
        layers.Dropout(dropout_rate, name="dropout"),
        layers.Dense(1, activation="sigmoid", bias_initializer=bias_init, name="dense_1")
    ], name="baseline_model")

    return model


def verify_baseline_parameters(model: keras.Model) -> Tuple[int, int, int]:
    """Verify total, trainable, and non-trainable parameter counts.

    Returns
    -------
    Tuple[int, int, int]
        (total_params, trainable_params, non_trainable_params)
    """
    trainable_count = int(np.sum([tf.size(w).numpy() for w in model.trainable_weights]))
    non_trainable_count = int(np.sum([tf.size(w).numpy() for w in model.non_trainable_weights]))
    total_count = trainable_count + non_trainable_count
    return total_count, trainable_count, non_trainable_count
