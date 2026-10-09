"""Data preprocessing pipeline implementing the paper's feature transformations."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class PreprocessedData:
    """Container for preprocessed train and test partitions."""
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    scaler: Optional[StandardScaler] = None

    def get_1d_cnn_tensors(self) -> Tuple[np.ndarray, np.ndarray]:
        """Reshape tabular 2D arrays to 3D tensors for 1D CNN models.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (X_train_3d, X_test_3d) with shape (n_samples, n_features, 1).
        """
        X_train_3d = np.expand_dims(self.X_train, axis=-1)
        X_test_3d = np.expand_dims(self.X_test, axis=-1)
        return X_train_3d, X_test_3d


def preprocess_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
    stratify: bool = True,
    scale_amount_only: bool = True,
    drop_time: bool = False,
    selected_features: Optional[List[str]] = None
) -> PreprocessedData:
    """Preprocess credit card fraud dataset with strict train-only scaler fitting.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset dataframe.
    test_size : float, default=0.20
        Fraction of dataset to reserve for test split (matches paper 56,962 test set).
    random_state : int, default=42
        Reproducibility seed.
    stratify : bool, default=True
        Whether to stratify target classes.
    scale_amount_only : bool, default=True
        If True, only standardizes 'Amount' using StandardScaler (paper's primary setup).
        If False, standardizes all predictor features.
    drop_time : bool, default=False
        If True, drops 'Time' column (resulting in 29 features matching Table 9 baseline).
    selected_features : list of str, optional
        Subset of predictor columns to select (for feature selection experiments).

    Returns
    -------
    PreprocessedData
        Dataclass containing preprocessed numpy arrays, feature names, and fitted scaler.
    """
    df_clean = df.copy()

    # Determine predictor features
    all_features = [c for c in df_clean.columns if c != "Class"]
    if drop_time and "Time" in all_features:
        all_features.remove("Time")

    if selected_features is not None:
        features_to_use = [f for f in selected_features if f in all_features]
    else:
        features_to_use = all_features

    X_raw = df_clean[features_to_use]
    y_raw = df_clean["Class"].values

    # Train / Test Split BEFORE fitting any transformation
    stratify_target = y_raw if stratify else None
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_raw,
        y_raw,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target
    )

    X_train = X_train_df.copy()
    X_test = X_test_df.copy()

    scaler: Optional[StandardScaler] = None

    if scale_amount_only:
        # Paper setting: standardise 'Amount' using StandardScaler
        if "Amount" in features_to_use:
            scaler = StandardScaler()
            X_train["Amount"] = scaler.fit_transform(X_train[["Amount"]].values)
            X_test["Amount"] = scaler.transform(X_test[["Amount"]].values)
    else:
        # Full standardization setting
        scaler = StandardScaler()
        X_train_vals = scaler.fit_transform(X_train.values)
        X_test_vals = scaler.transform(X_test.values)
        X_train = pd.DataFrame(X_train_vals, columns=features_to_use, index=X_train.index)
        X_test = pd.DataFrame(X_test_vals, columns=features_to_use, index=X_test.index)

    return PreprocessedData(
        X_train=X_train.values.astype(np.float32),
        X_test=X_test.values.astype(np.float32),
        y_train=y_train.astype(np.int32),
        y_test=y_test.astype(np.int32),
        feature_names=features_to_use,
        scaler=scaler
    )
