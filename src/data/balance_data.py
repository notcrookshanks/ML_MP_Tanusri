"""Dataset balancing via random undersampling as specified in Section IV-C-9."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class BalancedSplit:
    """Container for balanced experiment splits and sampling metadata."""
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    metadata: Dict[str, any]

    def get_1d_cnn_tensors(self) -> Tuple[np.ndarray, np.ndarray]:
        """Reshape tabular 2D arrays to 3D tensors for 1D CNN models."""
        return np.expand_dims(self.X_train, axis=-1), np.expand_dims(self.X_test, axis=-1)


def balance_training_set_leakage_free(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
    ratio: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, Dict[str, any]]:
    """Apply random undersampling to the training partition only (leakage-free).

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training target vector.
    random_state : int, default=42
        Sampling random seed.
    ratio : float, default=1.0
        Ratio of majority class to minority class (1.0 = 1:1 balance).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, dict]
        (X_balanced, y_balanced, sampling_metadata)
    """
    rng = np.random.RandomState(random_state)
    fraud_indices = np.where(y_train == 1)[0]
    legit_indices = np.where(y_train == 0)[0]

    n_fraud = len(fraud_indices)
    n_legit_to_sample = int(n_fraud * ratio)

    sampled_legit = rng.choice(legit_indices, size=n_legit_to_sample, replace=False)
    combined_indices = np.concatenate([fraud_indices, sampled_legit])
    rng.shuffle(combined_indices)

    metadata = {
        "method": "random_undersample_train_only",
        "random_state": random_state,
        "original_legit": len(legit_indices),
        "original_fraud": n_fraud,
        "balanced_legit": n_legit_to_sample,
        "balanced_fraud": n_fraud,
        "total_balanced": len(combined_indices),
        "sampled_indices": combined_indices.tolist()
    }

    return X_train[combined_indices], y_train[combined_indices], metadata


def create_presplit_balanced_dataset(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42
) -> BalancedSplit:
    """Create balanced dataset before splitting (reproducing paper's Table 7 balanced evaluation).

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.
    test_size : float, default=0.20
        Validation split fraction.
    random_state : int, default=42
        Random seed.

    Returns
    -------
    BalancedSplit
        Dataclass containing balanced train and validation sets.
    """
    rng = np.random.RandomState(random_state)

    fraud_df = df[df["Class"] == 1]
    legit_df = df[df["Class"] == 0]

    n_fraud = len(fraud_df)
    sampled_legit = legit_df.sample(n=n_fraud, random_state=random_state)

    balanced_df = pd.concat([fraud_df, sampled_legit], axis=0).sample(
        frac=1.0, random_state=random_state
    ).reset_index(drop=True)

    feature_cols = [c for c in df.columns if c != "Class"]
    X = balanced_df[feature_cols].copy()
    y = balanced_df["Class"].values

    # Standardize Amount if present
    if "Amount" in feature_cols:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        # To avoid data leakage across the balanced split, split first then scale
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        X_tr["Amount"] = scaler.fit_transform(X_tr[["Amount"]].values)
        X_te["Amount"] = scaler.transform(X_te[["Amount"]].values)
        X_train_arr = X_tr.values.astype(np.float32)
        X_test_arr = X_te.values.astype(np.float32)
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        X_train_arr = X_tr.values.astype(np.float32)
        X_test_arr = X_te.values.astype(np.float32)

    metadata = {
        "method": "random_undersample_presplit",
        "random_state": random_state,
        "original_total": len(df),
        "balanced_total": len(balanced_df),
        "train_samples": len(y_tr),
        "test_samples": len(y_te),
        "train_fraud": int((y_tr == 1).sum()),
        "test_fraud": int((y_te == 1).sum())
    }

    return BalancedSplit(
        X_train=X_train_arr,
        X_test=X_test_arr,
        y_train=y_tr.astype(np.int32),
        y_test=y_te.astype(np.int32),
        metadata=metadata
    )
