"""Comparison module evaluating reproduced metrics against paper's reported values."""

from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd

# Authoritative published benchmark metrics from paper tables
PAPER_REPORTED_TABLE_5 = {
    "decision_tree": {"accuracy": 99.93, "f1_score": 81.05},
    "knn": {"accuracy": 99.95, "f1_score": 85.71},
    "logistic_regression": {"accuracy": 99.91, "f1_score": 73.56},
    "svm": {"accuracy": 99.93, "f1_score": 77.71},
    "random_forest": {"accuracy": 99.92, "f1_score": 77.27},
    "xgboost": {"accuracy": 99.94, "f1_score": 84.49}
}

PAPER_REPORTED_TABLE_6 = {
    "cnn_epoch_35": {
        "accuracy": 99.9,
        "precision": 93.2,
        "recall": 77.5,
        "auc": 0.929,
        "prc": 0.816,
        "tp": 83,
        "fp": 6,
        "tn": 56849,
        "fn": 24
    },
    "cnn_epoch_14": {
        "accuracy": 99.8,
        "precision": 56.8,
        "recall": 81.3,
        "auc": 0.942,
        "prc": 0.741,
        "tp": 87,
        "fp": 66,
        "tn": 56789,
        "fn": 20
    }
}

PAPER_REPORTED_TABLE_7 = {
    "cnn_11_layers_ep20": {"train_acc": 93.2, "val_acc": 91.3},
    "cnn_13_layers_ep50": {"train_acc": 91.2, "val_acc": 91.3},
    "cnn_14_layers_ep100": {"train_acc": 96.34, "val_acc": 94.3},
    "cnn_17_layers_ep100": {"train_acc": 95.53, "val_acc": 94.7},
    "cnn_20_layers_ep100": {"train_acc": 94.92, "val_acc": 93.5},
    "cnn_14_layers_balanced": {"train_acc": 94.60, "val_acc": 95.80},
    "baseline_5_layers_ep20": {"train_acc": 90.7, "val_acc": 88.3}
}


def compare_ml_results(
    reproduced_results: List[Dict[str, any]]
) -> pd.DataFrame:
    """Compare reproduced classical ML results against published Table 5 values."""
    rows = []
    for res in reproduced_results:
        model_key = res.get("model", "").lower().replace(" ", "_")
        if model_key in PAPER_REPORTED_TABLE_5:
            pub = PAPER_REPORTED_TABLE_5[model_key]
            pub_acc = pub["accuracy"]
            pub_f1 = pub["f1_score"]
            rep_acc = res.get("accuracy_pct", 0.0)
            rep_f1 = res.get("f1_score_pct", 0.0)
            diff_acc = rep_acc - pub_acc
            diff_f1 = rep_f1 - pub_f1

            rows.append({
                "model": res.get("model"),
                "paper_accuracy": pub_acc,
                "reproduced_accuracy": rep_acc,
                "accuracy_diff": round(diff_acc, 4),
                "paper_f1_score": pub_f1,
                "reproduced_f1_score": rep_f1,
                "f1_score_diff": round(diff_f1, 4),
                "dataset_variant": res.get("dataset_variant", "standardized_amount")
            })

    return pd.DataFrame(rows)


def save_comparison_table(
    df: pd.DataFrame,
    output_path: Union[str, Path] = "reports/results/reported_vs_reproduced.csv"
) -> None:
    """Save comparison dataframe to CSV."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
