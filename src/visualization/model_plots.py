"""Model performance and training history visualizations."""

from pathlib import Path
from typing import Dict, List, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_classical_comparison(
    df_results: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = "reports/figures/figure_08_ml_comparison.png"
) -> plt.Figure:
    """Plot comparative analysis of ML models matching Figure 8 of the paper."""
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

    models = df_results["model"].tolist()
    x = np.arange(len(models))
    width = 0.35

    acc_vals = df_results["accuracy_pct"].tolist()
    f1_vals = df_results["f1_score_pct"].tolist()

    rects1 = ax.bar(x - width/2, acc_vals, width, label="Accuracy (%)", color="#1f77b4")
    rects2 = ax.bar(x + width/2, f1_vals, width, label="F1-Score (%)", color="#ff7f0e")

    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_title("Comparative Analysis of Machine Learning Algorithms", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in models], rotation=25, ha="right", fontsize=9)
    ax.set_ylim([0, 110])
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    for bar in rects1:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}%", (bar.get_x() + bar.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=7.5, rotation=45)
    for bar in rects2:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}%", (bar.get_x() + bar.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=7.5, rotation=45)

    plt.tight_layout()
    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight")
        fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    title: str = "Training and Validation History",
    output_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """Plot training and validation accuracy and loss over epochs matching Figures 14 and 15."""
    epochs = history.get("epoch", list(range(1, len(history.get("loss", [])) + 1)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)

    # Accuracy subplot
    if "accuracy" in history and "val_accuracy" in history:
        axes[0].plot(epochs, history["accuracy"], label="Train Accuracy", color="#1f77b4", lw=2)
        axes[0].plot(epochs, history["val_accuracy"], label="Val Accuracy", color="#ff7f0e", lw=2, linestyle="--")
        axes[0].set_title("Accuracy vs Epochs", fontsize=11, fontweight="bold")
        axes[0].set_xlabel("Epoch", fontsize=10)
        axes[0].set_ylabel("Accuracy", fontsize=10)
        axes[0].legend(loc="lower right", fontsize=9)
        axes[0].grid(True, linestyle=":", alpha=0.6)

    # Loss subplot
    if "loss" in history and "val_loss" in history:
        axes[1].plot(epochs, history["loss"], label="Train Loss", color="#1f77b4", lw=2)
        axes[1].plot(epochs, history["val_loss"], label="Val Loss", color="#d62728", lw=2, linestyle="--")
        axes[1].set_title("Loss vs Epochs", fontsize=11, fontweight="bold")
        axes[1].set_xlabel("Epoch", fontsize=10)
        axes[1].set_ylabel("Loss", fontsize=10)
        axes[1].legend(loc="upper right", fontsize=9)
        axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight")
        fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    return fig


def plot_depth_comparison(
    df_depth: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = "reports/figures/figure_17_cnn_depth_comparison.png"
) -> plt.Figure:
    """Plot CNN accuracy across layer depths matching Figure 17."""
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

    layers = df_depth["layers"].tolist()
    train_acc = df_depth["train_accuracy"].tolist()
    val_acc = df_depth["val_accuracy"].tolist()

    ax.plot(layers, train_acc, marker="o", lw=2, label="Train Accuracy", color="#1f77b4")
    ax.plot(layers, val_acc, marker="s", lw=2, label="Validation Accuracy", color="#ff7f0e", linestyle="--")

    ax.set_xlabel("Number of Layers", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("CNN Model Accuracy Over Number of Layers", fontsize=12, fontweight="bold")
    ax.set_xticks(layers)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight")
        fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    return fig


def plot_reported_vs_reproduced(
    df_comp: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = "reports/figures/figure_16_reported_vs_reproduced.png"
) -> plt.Figure:
    """Plot side-by-side comparison of paper reported vs reproduced metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    models = [m.replace("_", " ").title() for m in df_comp["model"]]
    x = np.arange(len(models))
    width = 0.35

    # Accuracy comparison
    axes[0].bar(x - width/2, df_comp["paper_accuracy"], width, label="Paper Reported", color="#2ca02c")
    axes[0].bar(x + width/2, df_comp["reproduced_accuracy"], width, label="Reproduced", color="#1f77b4")
    axes[0].set_title("Accuracy: Paper vs Reproduced", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Accuracy (%)", fontsize=10)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=30, ha="right", fontsize=9)
    axes[0].set_ylim([99.5, 100.1])
    axes[0].legend(loc="lower right", fontsize=9)
    axes[0].grid(axis="y", linestyle=":", alpha=0.5)

    # F1 comparison
    axes[1].bar(x - width/2, df_comp["paper_f1_score"], width, label="Paper Reported", color="#2ca02c")
    axes[1].bar(x + width/2, df_comp["reproduced_f1_score"], width, label="Reproduced", color="#ff7f0e")
    axes[1].set_title("F1-Score: Paper vs Reproduced", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("F1-Score (%)", fontsize=10)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=30, ha="right", fontsize=9)
    axes[1].set_ylim([0, 100])
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(axis="y", linestyle=":", alpha=0.5)

    plt.suptitle("Published vs Reproduced Benchmark Performance", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight")
        fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    return fig
