"""Unit tests verifying architectural parameters against paper Tables 8 and 9."""

import numpy as np
import pytest

from src.models.cnn_baseline import build_baseline_cnn, verify_baseline_parameters
from src.models.cnn_10_layers import build_cnn_10_layers, verify_cnn_10_parameters
from src.models.cnn_14_layers import build_cnn_14_layers
from src.models.cnn_17_layers import build_cnn_17_layers
from src.models.cnn_20_layers import build_cnn_20_layers


def test_baseline_parameter_counts_match_table_9():
    """Verify Baseline model matches Table 9: exactly 497 total and 497 trainable parameters."""
    # With 29 features (V1-V28 + Amount):
    # Dense(16): 29*16 + 16 = 480
    # Dropout: 0
    # Dense(1): 16*1 + 1 = 17
    # Total = 480 + 17 = 497
    model = build_baseline_cnn(input_dim=29)
    total, trainable, non_trainable = verify_baseline_parameters(model)

    assert total == 497, f"Expected 497 total params, got {total}"
    assert trainable == 497, f"Expected 497 trainable params, got {trainable}"
    assert non_trainable == 0, f"Expected 0 non-trainable params, got {non_trainable}"


def test_cnn_10_layers_parameter_counts_match_table_8():
    """Verify 10-Layer CNN matches Table 8: exactly 119,457 total, 119,265 trainable, 192 non-trainable."""
    # Input: (30, 1)
    # Conv1D(32, 2): (2*1*32)+32 = 96
    # BatchNorm(32): 4*32 = 128
    # Dropout: 0
    # Conv1D(64, 2): (2*32*64)+64 = 4160
    # BatchNorm(64): 4*64 = 256
    # Dropout: 0
    # Flatten: (28*64) = 1792
    # Dense(64): 1792*64 + 64 = 114752
    # Dropout: 0
    # Dense(1): 64*1 + 1 = 65
    # Total = 96 + 128 + 4160 + 256 + 114752 + 65 = 119,457
    # Trainable = 119,457 - (128/2 + 256/2) = 119,265
    # Non-trainable = 192
    model = build_cnn_10_layers(input_dim=30)
    total, trainable, non_trainable = verify_cnn_10_parameters(model)

    assert total == 119457, f"Expected 119,457 total params, got {total}"
    assert trainable == 119265, f"Expected 119,265 trainable params, got {trainable}"
    assert non_trainable == 192, f"Expected 192 non-trainable params, got {non_trainable}"


def test_proposed_cnn_layer_counts():
    """Verify proposed architectures have exactly 14, 17, and 20 sequential layers."""
    model_14 = build_cnn_14_layers(input_dim=30)
    assert len(model_14.layers) == 14, f"Expected 14 layers, got {len(model_14.layers)}"

    model_17 = build_cnn_17_layers(input_dim=30)
    assert len(model_17.layers) == 17, f"Expected 17 layers, got {len(model_17.layers)}"

    model_20 = build_cnn_20_layers(input_dim=30)
    assert len(model_20.layers) == 20, f"Expected 20 layers, got {len(model_20.layers)}"
