"""ROC and Precision-Recall curve generation matching paper Figures 10 and 14."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score


def plot_roc_curves(
    models_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Receiver Operating Characteristic (ROC) Curves"
) -> plt.Figure:
    """Plot overlaid ROC curves for multiple models."""
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

    for model_name, (y_true, y_prob) in models_predictions.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--", label="Chance (AUC = 0.50)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight")
        fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")

    return fig


def plot_pr_curves(
    models_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Precision-Recall Curves"
) -> plt.Figure:
    """Plot overlaid Precision-Recall curves matching Figure 10."""
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

    for model_name, (y_true, y_prob) in models_predictions.items():
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(recall, precision, lw=2, label=f"{model_name} (AP = {ap:.4f})")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight")
        fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")

    return fig
