"""Unit tests for dataset schema, integrity, and validation rules."""

import numpy as np
import pandas as pd
import pytest

from src.data.validate_data import (
    EXPECTED_COLUMNS,
    EXPECTED_TOTAL_ROWS,
    EXPECTED_FRAUD_COUNT,
    EXPECTED_LEGIT_COUNT,
    DataValidationError,
    validate_dataset
)


def create_synthetic_credit_data(n_rows: int = 100, n_fraud: int = 10) -> pd.DataFrame:
    """Helper to generate a valid synthetic credit card dataset."""
    rng = np.random.RandomState(42)
    data = {}
    data["Time"] = np.linspace(0, 1000, n_rows)
    for i in range(1, 29):
        data[f"V{i}"] = rng.normal(0, 1, n_rows)
    data["Amount"] = rng.exponential(50, n_rows)

    classes = np.zeros(n_rows, dtype=int)
    fraud_indices = rng.choice(n_rows, size=n_fraud, replace=False)
    classes[fraud_indices] = 1
    data["Class"] = classes

    return pd.DataFrame(data)


def test_synthetic_data_validation_success():
    """Verify that a synthetic dataset matching the schema passes non-strict validation."""
    df_synth = create_synthetic_credit_data(n_rows=200, n_fraud=10)
    report = validate_dataset(df_synth, strict_row_count=False)
    assert report["is_valid"] is True
    assert report["statistics"]["class_distribution"]["fraudulent"] == 10
    assert report["statistics"]["class_distribution"]["legitimate"] == 190


def test_missing_column_raises_validation_error():
    """Ensure missing columns trigger DataValidationError."""
    df_synth = create_synthetic_credit_data(n_rows=50, n_fraud=5)
    df_bad = df_synth.drop(columns=["V1", "Amount"])
    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate_dataset(df_bad, strict_row_count=False)


def test_null_values_raise_validation_error():
    """Ensure null/NaN values are caught and flagged."""
    df_synth = create_synthetic_credit_data(n_rows=50, n_fraud=5)
    df_synth.loc[0, "V10"] = np.nan
    with pytest.raises(DataValidationError, match="null/missing values"):
        validate_dataset(df_synth, strict_row_count=False)


def test_invalid_class_values_raise_error():
    """Ensure labels other than 0 and 1 are rejected."""
    df_synth = create_synthetic_credit_data(n_rows=50, n_fraud=5)
    df_synth.loc[0, "Class"] = 2
    with pytest.raises(DataValidationError, match="Invalid target classes"):
        validate_dataset(df_synth, strict_row_count=False)
