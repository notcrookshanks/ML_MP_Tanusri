"""Unit tests for preprocessing and leakage prevention."""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import preprocess_data
from tests.test_data_validation import create_synthetic_credit_data


def test_preprocessing_split_shapes():
    """Verify train and test split sizes conform to test_size parameter."""
    df_synth = create_synthetic_credit_data(n_rows=100, n_fraud=10)
    prep = preprocess_data(df_synth, test_size=0.20, random_state=42)

    assert prep.X_train.shape[0] == 80
    assert prep.X_test.shape[0] == 20
    assert len(prep.y_train) == 80
    assert len(prep.y_test) == 20
    assert prep.X_train.shape[1] == 30  # 30 predictors


def test_preprocessing_scaler_fit_on_train_only():
    """Verify scaler is fit exclusively on train set to prevent leakage."""
    df_synth = create_synthetic_credit_data(n_rows=100, n_fraud=10)
    # Add an extreme outlier only in the test portion
    prep = preprocess_data(df_synth, test_size=0.20, random_state=42, scale_amount_only=True)

    amount_idx = prep.feature_names.index("Amount")
    train_amount_mean = np.mean(prep.X_train[:, amount_idx])
    train_amount_std = np.std(prep.X_train[:, amount_idx])

    # Train standardized amount should have mean ~ 0 and std ~ 1
    assert abs(train_amount_mean) < 1e-4
    assert abs(train_amount_std - 1.0) < 1e-4


def test_1d_cnn_tensor_reshaping():
    """Verify 3D reshaping produces shape (samples, features, 1)."""
    df_synth = create_synthetic_credit_data(n_rows=50, n_fraud=5)
    prep = preprocess_data(df_synth, test_size=0.20, random_state=42)
    X_tr_3d, X_te_3d = prep.get_1d_cnn_tensors()

    assert X_tr_3d.ndim == 3
    assert X_tr_3d.shape == (40, 30, 1)
    assert X_te_3d.shape == (10, 30, 1)


def test_drop_time_option():
    """Verify dropping 'Time' produces 29 features."""
    df_synth = create_synthetic_credit_data(n_rows=50, n_fraud=5)
    prep = preprocess_data(df_synth, test_size=0.20, drop_time=True)

    assert prep.X_train.shape[1] == 29
    assert "Time" not in prep.feature_names
