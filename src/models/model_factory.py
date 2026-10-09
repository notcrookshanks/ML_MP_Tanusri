"""Unified model factory for classical and deep learning architectures."""

from typing import Any, Dict, List, Optional
import tensorflow as tf
from tensorflow import keras

from src.models.classical_models import (
    build_decision_tree,
    build_elm,
    build_knn,
    build_logistic_regression,
    build_logistic_regression_with_random_search,
    build_random_forest,
    build_svm,
    build_xgboost
)
from src.models.cnn_baseline import build_baseline_cnn
from src.models.cnn_10_layers import build_cnn_10_layers
from src.models.cnn_14_layers import build_cnn_14_layers
from src.models.cnn_17_layers import build_cnn_17_layers
from src.models.cnn_20_layers import build_cnn_20_layers


CLASSICAL_MODELS = [
    "decision_tree",
    "knn",
    "logistic_regression",
    "logistic_regression_random_search",
    "svm",
    "random_forest",
    "xgboost",
    "elm"
]

DL_MODELS = [
    "baseline_nn",
    "cnn_10_layers",
    "cnn_14_layers",
    "cnn_17_layers",
    "cnn_20_layers"
]


def create_classical_model(
    model_name: str,
    random_state: int = 42,
    **kwargs: Any
) -> Any:
    """Instantiate a classical ML classifier by name."""
    name = model_name.lower()
    if name == "decision_tree":
        return build_decision_tree(
            max_depth=kwargs.get("max_depth", 4),
            criterion=kwargs.get("criterion", "entropy"),
            random_state=random_state
        )
    elif name == "knn":
        return build_knn(
            n_neighbors=kwargs.get("n_neighbors", 5),
            weights=kwargs.get("weights", "uniform")
        )
    elif name == "logistic_regression":
        return build_logistic_regression(
            C=kwargs.get("C", 1.0),
            random_state=random_state,
            max_iter=kwargs.get("max_iter", 1000)
        )
    elif name == "logistic_regression_random_search":
        return build_logistic_regression_with_random_search(
            param_distributions=kwargs.get("param_distributions", None),
            n_iter=kwargs.get("n_iter", 6),
            cv=kwargs.get("cv", 5),
            random_state=random_state,
            max_iter=kwargs.get("max_iter", 1000)
        )
    elif name == "svm":
        return build_svm(
            C=kwargs.get("C", 1.0),
            kernel=kwargs.get("kernel", "rbf"),
            probability=kwargs.get("probability", True),
            random_state=random_state
        )
    elif name == "random_forest":
        return build_random_forest(
            n_estimators=kwargs.get("n_estimators", 100),
            criterion=kwargs.get("criterion", "entropy"),
            random_state=random_state
        )
    elif name == "xgboost":
        return build_xgboost(
            n_estimators=kwargs.get("n_estimators", 100),
            learning_rate=kwargs.get("learning_rate", 0.1),
            max_depth=kwargs.get("max_depth", 6),
            eval_metric=kwargs.get("eval_metric", "logloss"),
            random_state=random_state
        )
    elif name == "elm":
        return build_elm(
            n_hidden_nodes=kwargs.get("n_hidden_nodes", 50),
            activation=kwargs.get("activation", "relu"),
            c_regularization=kwargs.get("c_regularization", 1.0),
            random_state=random_state
        )
    else:
        raise ValueError(f"Unknown classical model '{model_name}'. Available: {CLASSICAL_MODELS}")


def create_dl_model(
    architecture: str,
    input_dim: int = 30,
    compile_model: bool = True,
    learning_rate: float = 0.001,
    output_bias: Optional[float] = None
) -> keras.Model:
    """Instantiate a deep learning model by architecture name."""
    arch = architecture.lower()
    if arch in ["baseline", "baseline_nn", "baseline_model"]:
        model = build_baseline_cnn(
            input_dim=input_dim,
            hidden_units=16,
            dropout_rate=0.5,
            output_bias=output_bias
        )
    elif arch in ["cnn_10", "cnn_10_layers", "cnn_sequential"]:
        model = build_cnn_10_layers(input_dim=input_dim)
    elif arch in ["cnn_14", "cnn_14_layers"]:
        model = build_cnn_14_layers(input_dim=input_dim)
    elif arch in ["cnn_17", "cnn_17_layers"]:
        model = build_cnn_17_layers(input_dim=input_dim)
    elif arch in ["cnn_20", "cnn_20_layers"]:
        model = build_cnn_20_layers(input_dim=input_dim)
    else:
        raise ValueError(f"Unknown DL architecture '{architecture}'. Available: {DL_MODELS}")

    if compile_model:
        metrics = [
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc", curve="ROC"),
            keras.metrics.AUC(name="prc", curve="PR")
        ]
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=metrics
        )

    return model
