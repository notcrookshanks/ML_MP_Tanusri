"""Visualizations for Exploratory Data Analysis matching paper figures."""

from pathlib import Path
from typing import Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_class_distribution(
    df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = "reports/figures/figure_05_class_distribution.png"
) -> plt.Figure:
    """Plot class distribution matching Figure 5 of the paper."""
    counts = df["Class"].value_counts()
    legit_count = counts.get(0, 0)
    fraud_count = counts.get(1, 0)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    bars = ax.bar(["0: Verified", "1: Fraud"], [legit_count, fraud_count], color=["#1f77b4", "#d62728"], width=0.5)

    ax.set_ylabel("Number of Transactions", fontsize=11)
    ax.set_title("Class Distributions\n0: Verified || 1: Fraud", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    # Annotate counts
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:,}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight")
        fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    return fig


def plot_amount_statistics(
    df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = "reports/figures/figure_07_amount_statistics.png"
) -> plt.Figure:
    """Plot case amount statistics matching Figure 7 of the paper."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)

    fraud_amounts = df[df["Class"] == 1]["Amount"]
    legit_amounts = df[df["Class"] == 0]["Amount"]

    # Fraud Amount distribution
    axes[0].hist(fraud_amounts, bins=30, color="#d62728", edgecolor="black", alpha=0.8)
    axes[0].set_title("FRAUD CASE AMOUNT STATS", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Amount (€)", fontsize=10)
    axes[0].set_ylabel("Count", fontsize=10)
    axes[0].grid(True, linestyle=":", alpha=0.5)

    # Non-fraud Amount distribution (log scale for visual clarity)
    axes[1].hist(legit_amounts, bins=50, color="#1f77b4", edgecolor="black", alpha=0.8)
    axes[1].set_yscale("log")
    axes[1].set_title("NON-FRAUD CASE AMOUNT STATS (Log Scale)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Amount (€)", fontsize=10)
    axes[1].set_ylabel("Count (Log Scale)", fontsize=10)
    axes[1].grid(True, linestyle=":", alpha=0.5)

    plt.suptitle("CASE AMOUNT STATISTICS", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight")
        fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    return fig


def plot_correlation_matrix(
    df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = "reports/figures/feature_correlation_matrix.png"
) -> plt.Figure:
    """Plot correlation matrix across all features."""
    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
    corr = df.corr()
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, cbar=True, square=True)
    ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()

    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight")
        fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    return fig


def plot_feature_distributions_by_class(
    df: pd.DataFrame,
    features: Optional[list] = None,
    output_path: Optional[Union[str, Path]] = "reports/figures/figure_11_12_feature_distributions.png"
) -> plt.Figure:
    """Plot distribution of key discriminating features matching Figures 11 and 12."""
    if features is None:
        features = ["V4", "V11", "V12", "V14"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=300)
    axes = axes.flatten()

    for i, feat in enumerate(features):
        ax = axes[i]
        sns.kdeplot(df[df["Class"] == 0][feat], label="Non-Fraud (0)", color="#1f77b4", ax=ax, fill=True, alpha=0.3)
        sns.kdeplot(df[df["Class"] == 1][feat], label="Fraud (1)", color="#d62728", ax=ax, fill=True, alpha=0.3)
        ax.set_title(f"Distribution of {feat}", fontsize=11, fontweight="bold")
        ax.set_xlabel(feat, fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.5)

    plt.suptitle("Feature Distributions by Class (Positive vs Negative)", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight")
        fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    return fig
