"""Feature ranking and selection experiments."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif


def compute_feature_rankings(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    method: str = "random_forest",
    random_state: int = 42
) -> pd.DataFrame:
    """Compute feature importance rankings on training data.

    Parameters
    ----------
    X : np.ndarray
        Training feature matrix.
    y : np.ndarray
        Training labels.
    feature_names : list of str
        Names of the features.
    method : str, default="random_forest"
        Method to use: 'random_forest' (Gini importance) or 'anova_f' or 'mutual_info'.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['feature', 'score', 'rank'] sorted by importance descending.
    """
    if method == "random_forest":
        rf = RandomForestClassifier(
            n_estimators=100,
            criterion="entropy",
            random_state=random_state,
            n_jobs=-1
        )
        rf.fit(X, y)
        scores = rf.feature_importances_
    elif method == "anova_f":
        selector = SelectKBest(score_func=f_classif, k="all")
        selector.fit(X, y)
        scores = selector.scores_
        scores = np.nan_to_num(scores, nan=0.0)
    elif method == "mutual_info":
        scores = mutual_info_classif(X, y, random_state=random_state)
    else:
        raise ValueError(f"Unknown feature ranking method: '{method}'")

    df_rank = pd.DataFrame({
        "feature": feature_names,
        "score": scores
    })
    df_rank = df_rank.sort_values(by="score", ascending=False).reset_index(drop=True)
    df_rank["rank"] = df_rank.index + 1
    return df_rank


def select_top_features(
    ranking_df: pd.DataFrame,
    top_k: int = 15
) -> List[str]:
    """Select the top K feature names from a ranking DataFrame."""
    return ranking_df["feature"].iloc[:top_k].tolist()


def save_feature_rankings(
    ranking_df: pd.DataFrame,
    output_path: Union[str, Path] = "reports/results/feature_ranking.csv"
) -> None:
    """Save feature rankings to CSV."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ranking_df.to_csv(p, index=False)
