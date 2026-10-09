"""Extreme Learning Machine (ELM) classifier implementation."""

from typing import Optional, Union
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array


class ELMClassifier(BaseEstimator, ClassifierMixin):
    """Extreme Learning Machine for binary classification.

    Implements a single-hidden-layer feedforward network (SLFN) where
    input weights and biases are randomly initialized, and output weights
    are analytically solved in a single step using Moore-Penrose pseudoinverse,
    as mathematically described in Section III-B-i (Equations 1-2).

    Parameters
    ----------
    n_hidden_nodes : int, default=50
        Number of hidden neurons.
    activation : str, default='relu'
        Hidden layer activation function: 'relu', 'sigmoid', or 'tanh'.
    c_regularization : float, default=1.0
        Ridge regression regularization parameter (L2 penalty).
    random_state : int, default=42
        Seed for random weight initialization.
    """

    def __init__(
        self,
        n_hidden_nodes: int = 50,
        activation: str = "relu",
        c_regularization: float = 1.0,
        random_state: int = 42
    ):
        self.n_hidden_nodes = n_hidden_nodes
        self.activation = activation
        self.c_regularization = c_regularization
        self.random_state = random_state

    def _activate(self, H: np.ndarray) -> np.ndarray:
        """Apply the chosen activation function."""
        if self.activation == "relu":
            return np.maximum(H, 0.0)
        elif self.activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(H, -25.0, 25.0)))
        elif self.activation == "tanh":
            return np.tanh(H)
        else:
            raise ValueError(f"Unsupported activation: '{self.activation}'")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ELMClassifier":
        """Fit the ELM classifier using analytic closed-form solver."""
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("ELMClassifier currently configured for binary classification.")

        n_samples, n_features = X.shape

        rng = np.random.RandomState(self.random_state)
        # Random input weights: (n_features, n_hidden)
        self.input_weights_ = rng.normal(size=(n_features, self.n_hidden_nodes))
        # Random biases: (1, n_hidden)
        self.biases_ = rng.normal(size=(1, self.n_hidden_nodes))

        # Hidden layer feature map H
        H = self._activate(np.dot(X, self.input_weights_) + self.biases_)

        # Target matrix T in {-1, 1} or {0, 1}
        # Using two-column one-hot encoding for calibrated probabilities
        T = np.zeros((n_samples, 2), dtype=np.float64)
        for i, val in enumerate(y):
            T[i, int(val)] = 1.0

        # Output weight solving: beta = (H^T H + I/C)^(-1) H^T T
        HTH = np.dot(H.T, H)
        reg = np.eye(self.n_hidden_nodes) / max(self.c_regularization, 1e-8)
        self.output_weights_ = np.dot(np.linalg.pinv(HTH + reg), np.dot(H.T, T))

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for X."""
        check_is_fitted(self, ["input_weights_", "biases_", "output_weights_"])
        X = check_array(X)

        H = self._activate(np.dot(X, self.input_weights_) + self.biases_)
        raw_scores = np.dot(H, self.output_weights_)

        # Softmax normalization over output columns
        exp_scores = np.exp(raw_scores - np.max(raw_scores, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for X."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
