"""Unit tests for feature selection and ranking algorithms."""

import numpy as np
import pytest

from src.features.feature_ranking import compute_feature_rankings, select_top_features


def test_feature_rankings_random_forest():
    """Verify feature rankings return sorted DataFrame with valid ranks and positive scores."""
    rng = np.random.RandomState(42)
    n_samples = 100
    n_features = 10
    X = rng.normal(size=(n_samples, n_features))
    # Make feature 0 strongly correlated with y
    y = (X[:, 0] > 0).astype(int)
    feature_names = [f"F_{i}" for i in range(n_features)]

    ranking_df = compute_feature_rankings(X, y, feature_names, method="random_forest", random_state=42)

    assert len(ranking_df) == n_features
    assert "feature" in ranking_df.columns
    assert "score" in ranking_df.columns
    assert "rank" in ranking_df.columns
    assert ranking_df["rank"].iloc[0] == 1
    # Most informative feature should be top ranked
    assert ranking_df["feature"].iloc[0] == "F_0"


def test_feature_rankings_anova_f():
    """Verify ANOVA F-test scoring ranking works correctly."""
    rng = np.random.RandomState(42)
    X = rng.normal(size=(100, 5))
    y = (X[:, 2] > 0).astype(int)
    feature_names = [f"F_{i}" for i in range(5)]

    ranking_df = compute_feature_rankings(X, y, feature_names, method="anova_f")
    assert len(ranking_df) == 5
    assert ranking_df["feature"].iloc[0] == "F_2"


def test_select_top_features():
    """Verify top-K selection returns requested number of features."""
    rng = np.random.RandomState(42)
    X = rng.normal(size=(50, 8))
    y = (X[:, 1] > 0).astype(int)
    feature_names = [f"F_{i}" for i in range(8)]
    ranking_df = compute_feature_rankings(X, y, feature_names)

    top_3 = select_top_features(ranking_df, top_k=3)
    assert len(top_3) == 3
    assert all(f in feature_names for f in top_3)
