"""Traditional Machine Learning model implementations matching paper specifications."""

from typing import Any, Dict, Optional
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.models.elm import ELMClassifier


def build_decision_tree(
    max_depth: int = 4,
    criterion: str = "entropy",
    random_state: int = 42
) -> DecisionTreeClassifier:
    """Build Decision Tree classifier with paper settings (max_depth=4, criterion='entropy')."""
    return DecisionTreeClassifier(
        max_depth=max_depth,
        criterion=criterion,
        random_state=random_state
    )


def build_knn(
    n_neighbors: int = 5,
    weights: str = "uniform"
) -> KNeighborsClassifier:
    """Build K-Nearest Neighbors classifier with paper settings (n_neighbors=5)."""
    return KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        n_jobs=-1
    )


def build_logistic_regression(
    C: float = 1.0,
    random_state: int = 42,
    max_iter: int = 1000
) -> LogisticRegression:
    """Build Logistic Regression classifier."""
    return LogisticRegression(
        C=C,
        random_state=random_state,
        max_iter=max_iter,
        solver="lbfgs"
    )


def build_logistic_regression_with_random_search(
    param_distributions: Optional[Dict[str, Any]] = None,
    n_iter: int = 6,
    cv: int = 5,
    random_state: int = 42,
    max_iter: int = 1000
) -> RandomizedSearchCV:
    """Build Logistic Regression with RandomizedSearchCV over parameter C as described in paper."""
    if param_distributions is None:
        param_distributions = {"C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}

    base_lr = LogisticRegression(
        random_state=random_state,
        max_iter=max_iter,
        solver="lbfgs"
    )

    search = RandomizedSearchCV(
        estimator=base_lr,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring="f1",
        random_state=random_state,
        n_jobs=-1
    )
    return search


def build_svm(
    C: float = 1.0,
    kernel: str = "rbf",
    probability: bool = True,
    random_state: int = 42
) -> SVC:
    """Build Support Vector Machine classifier."""
    return SVC(
        C=C,
        kernel=kernel,
        probability=probability,
        random_state=random_state
    )


def build_random_forest(
    n_estimators: int = 100,
    criterion: str = "entropy",
    random_state: int = 42
) -> RandomForestClassifier:
    """Build Random Forest ensemble classifier."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        random_state=random_state,
        n_jobs=-1
    )


def build_xgboost(
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 6,
    eval_metric: str = "logloss",
    random_state: int = 42
) -> XGBClassifier:
    """Build XGBoost classifier with serviceable defaults."""
    return XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        eval_metric=eval_metric,
        random_state=random_state,
        n_jobs=-1
    )


def build_elm(
    n_hidden_nodes: int = 50,
    activation: str = "relu",
    c_regularization: float = 1.0,
    random_state: int = 42
) -> ELMClassifier:
    """Build Extreme Learning Machine classifier."""
    return ELMClassifier(
        n_hidden_nodes=n_hidden_nodes,
        activation=activation,
        c_regularization=c_regularization,
        random_state=random_state
    )
