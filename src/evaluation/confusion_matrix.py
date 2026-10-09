"""Confusion matrix computation and plotting utilities."""

from pathlib import Path
from typing import Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    output_path: Optional[Union[str, Path]] = None,
    class_names: Optional[list] = None
) -> plt.Figure:
    """Plot a publication-quality confusion matrix heatmap matching Figure 6."""
    if class_names is None:
        class_names = ["Legitimate (0)", "Fraudulent (1)"]

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax
    )
    ax.set_title(f"Confusion Matrix: {model_name}", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=10, labelpad=8)
    ax.set_ylabel("True Label", fontsize=10, labelpad=8)
    plt.tight_layout()

    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight")
        # Also save PDF if desired
        pdf_path = p.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")

    return fig
