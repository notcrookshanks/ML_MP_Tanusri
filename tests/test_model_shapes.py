"""Unit tests verifying model tensor input/output shapes and binary classification behavior."""

import numpy as np
import pytest

from src.models.classical_models import (
    build_decision_tree,
    build_elm,
    build_knn,
    build_logistic_regression,
    build_random_forest,
    build_xgboost
)
from src.models.cnn_baseline import build_baseline_cnn
from src.models.cnn_10_layers import build_cnn_10_layers
from src.models.cnn_14_layers import build_cnn_14_layers
from src.models.cnn_17_layers import build_cnn_17_layers
from src.models.cnn_20_layers import build_cnn_20_layers


def test_classical_models_output_shapes():
    """Verify classical models produce 1D binary predictions and 2D valid probabilities."""
    rng = np.random.RandomState(42)
    X = rng.normal(size=(40, 10))
    y = (rng.rand(40) > 0.5).astype(int)

    models = [
        build_decision_tree(random_state=42),
        build_knn(n_neighbors=3),
        build_logistic_regression(random_state=42),
        build_random_forest(n_estimators=10, random_state=42),
        build_xgboost(n_estimators=10, random_state=42),
        build_elm(n_hidden_nodes=20, random_state=42)
    ]

    for model in models:
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (40,)
        assert set(np.unique(preds)).issubset({0, 1})

        probs = model.predict_proba(X)
        assert probs.shape == (40, 2)
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
        assert np.allclose(np.sum(probs, axis=1), 1.0)


def test_dl_models_output_shapes():
    """Verify deep learning models output a single probability in [0, 1] per sample."""
    n_samples = 16
    input_dim = 30
    X_2d = np.random.normal(size=(n_samples, input_dim)).astype(np.float32)
    X_3d = np.expand_dims(X_2d, axis=-1)

    # Baseline MLP (expects 2D)
    model_bl = build_baseline_cnn(input_dim=input_dim)
    out_bl = model_bl(X_2d, training=False).numpy()
    assert out_bl.shape == (n_samples, 1)
    assert np.all(out_bl >= 0.0) and np.all(out_bl <= 1.0)

    # 1D CNNs (expect 3D: (samples, steps, channels))
    cnn_models = [
        build_cnn_10_layers(input_dim=input_dim),
        build_cnn_14_layers(input_dim=input_dim),
        build_cnn_17_layers(input_dim=input_dim),
        build_cnn_20_layers(input_dim=input_dim)
    ]

    for model in cnn_models:
        out = model(X_3d, training=False).numpy()
        assert out.shape == (n_samples, 1)
        assert np.all(out >= 0.0) and np.all(out <= 1.0)
