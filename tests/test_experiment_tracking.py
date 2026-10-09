"""Unit tests for experiment tracking, configuration serialization, and balancing behavior."""

import numpy as np
import yaml
import pytest

from src.data.balance_data import (
    balance_training_set_leakage_free,
    create_presplit_balanced_dataset
)
from tests.test_data_validation import create_synthetic_credit_data


def test_leakage_free_undersampling():
    """Verify that undersampling on training data produces exact 1:1 balance."""
    X_train = np.random.normal(size=(200, 10))
    # 180 legit (0), 20 fraud (1)
    y_train = np.array([0] * 180 + [1] * 20)

    X_bal, y_bal, meta = balance_training_set_leakage_free(
        X_train, y_train, random_state=42, ratio=1.0
    )

    assert len(y_bal) == 40
    assert int((y_bal == 1).sum()) == 20
    assert int((y_bal == 0).sum()) == 20
    assert meta["balanced_fraud"] == 20
    assert meta["balanced_legit"] == 20


def test_presplit_balancing():
    """Verify pre-split balanced dataset creation."""
    df_synth = create_synthetic_credit_data(n_rows=200, n_fraud=20)
    split = create_presplit_balanced_dataset(df_synth, test_size=0.20, random_state=42)

    # 20 fraud -> balanced to 20 fraud + 20 legit = 40 samples total
    assert split.metadata["balanced_total"] == 40
    total_split_samples = len(split.y_train) + len(split.y_test)
    assert total_split_samples == 40


def test_config_yaml_serialization(tmp_path):
    """Verify YAML configuration loading and saving."""
    cfg = {
        "seed": 42,
        "models": ["decision_tree", "cnn_14_layers"],
        "epochs": 100
    }
    cfg_file = tmp_path / "test_config.yaml"
    with open(cfg_file, "w") as f:
        yaml.dump(cfg, f)

    with open(cfg_file, "r") as f:
        loaded = yaml.safe_load(f)

    assert loaded["seed"] == 42
    assert "cnn_14_layers" in loaded["models"]
