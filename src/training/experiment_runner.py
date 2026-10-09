"""Central Experiment Runner executing all reproduction experiments."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import yaml

from src.data.load_data import load_raw_data
from src.data.validate_data import validate_dataset, save_validation_report
from src.data.preprocess import preprocess_data
from src.data.balance_data import create_presplit_balanced_dataset, balance_training_set_leakage_free
from src.features.feature_ranking import compute_feature_rankings, save_feature_rankings
from src.training.train_classical import train_and_evaluate_classical_model
from src.training.train_cnn import train_and_evaluate_dl_model
from src.evaluation.compare_results import compare_ml_results, save_comparison_table
from src.evaluation.confusion_matrix import plot_confusion_matrix
from src.evaluation.curves import plot_roc_curves, plot_pr_curves
from src.visualization.dataset_plots import (
    plot_class_distribution,
    plot_amount_statistics,
    plot_correlation_matrix,
    plot_feature_distributions_by_class
)
from src.visualization.model_plots import (
    plot_classical_comparison,
    plot_training_history,
    plot_depth_comparison,
    plot_reported_vs_reproduced
)


class ExperimentRunner:
    """Orchestrates end-to-end execution of all paper reproduction experiments."""

    def __init__(self, config_path: str = "configs/base.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.seed = self.config.get("seed", 42)
        self.raw_data_path = self.config["data"]["raw_path"]
        self.reports_dir = Path(self.config["output"]["reports_dir"])
        self.figures_dir = Path(self.config["output"]["figures_dir"])
        self.results_dir = Path(self.config["output"]["results_dir"])

        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.df_raw: Optional[pd.DataFrame] = None
        self.preprocessed = None
        self.validation_report: Optional[Dict[str, Any]] = None

    def load_and_validate(self, strict: bool = True) -> pd.DataFrame:
        """Load and validate the raw dataset."""
        print(f"Loading raw dataset from {self.raw_data_path}...")
        self.df_raw = load_raw_data(self.raw_data_path)
        print("Validating dataset integrity and schema...")
        self.validation_report = validate_dataset(self.df_raw, strict_row_count=strict)
        save_validation_report(self.validation_report, self.results_dir / "data_validation_report.json")
        print("Data validation successful.")
        return self.df_raw

    def generate_eda_figures(self) -> None:
        """Generate Exploratory Data Analysis figures matching paper Figures 5, 7, 11, 12."""
        if self.df_raw is None:
            self.load_and_validate()

        print("Generating EDA figures...")
        plot_class_distribution(self.df_raw, self.figures_dir / "figure_05_class_distribution.png")
        plot_amount_statistics(self.df_raw, self.figures_dir / "figure_07_amount_statistics.png")
        plot_correlation_matrix(self.df_raw, self.figures_dir / "feature_correlation_matrix.png")
        plot_feature_distributions_by_class(self.df_raw, output_path=self.figures_dir / "figure_11_12_feature_distributions.png")
        print("EDA figures generated.")

    def run_feature_ranking(self) -> pd.DataFrame:
        """Run feature ranking experiment and save ranking report."""
        if self.preprocessed is None:
            self.preprocessed = preprocess_data(
                self.df_raw,
                test_size=self.config["split"]["test_size"],
                random_state=self.seed,
                stratify=self.config["split"]["stratify"],
                scale_amount_only=True
            )

        print("Computing feature rankings via Random Forest...")
        rank_df = compute_feature_rankings(
            self.preprocessed.X_train,
            self.preprocessed.y_train,
            feature_names=self.preprocessed.feature_names,
            method="random_forest",
            random_state=self.seed
        )
        save_feature_rankings(rank_df, self.results_dir / "feature_ranking.csv")
        return rank_df

    def run_classical_ml_experiments(
        self,
        svm_max_samples: Optional[int] = 30000
    ) -> pd.DataFrame:
        """Execute all classical ML experiments (Table 5)."""
        if self.df_raw is None:
            self.load_and_validate()
        if self.preprocessed is None:
            self.preprocessed = preprocess_data(
                self.df_raw,
                test_size=self.config["split"]["test_size"],
                random_state=self.seed,
                stratify=self.config["split"]["stratify"],
                scale_amount_only=True
            )

        models = [
            "decision_tree",
            "knn",
            "logistic_regression",
            "random_forest",
            "svm",
            "xgboost",
            "elm"
        ]

        print(f"Running classical ML experiments on models: {models}...")
        results = []
        roc_dict = {}
        pr_dict = {}

        for model_name in models:
            print(f"  Training {model_name}...")
            res = train_and_evaluate_classical_model(
                model_name=model_name,
                X_train=self.preprocessed.X_train,
                y_train=self.preprocessed.y_train,
                X_test=self.preprocessed.X_test,
                y_test=self.preprocessed.y_test,
                random_state=self.seed,
                dataset_variant="standardized_amount",
                svm_max_samples=svm_max_samples
            )

            # Generate individual confusion matrix plot
            cm = np.array([
                [res["true_negatives"], res["false_positives"]],
                [res["false_negatives"], res["true_positives"]]
            ])
            plot_confusion_matrix(
                cm,
                model_name=model_name.replace("_", " ").title(),
                output_path=self.figures_dir / f"confusion_matrix_{model_name}.png"
            )

            # Store for curves
            if res["y_prob"] is not None:
                display_name = model_name.replace("_", " ").title()
                roc_dict[display_name] = (self.preprocessed.y_test, res["y_prob"])
                pr_dict[display_name] = (self.preprocessed.y_test, res["y_prob"])

            clean_res = {k: v for k, v in res.items() if k not in ["y_pred", "y_prob"]}
            results.append(clean_res)

        df_ml = pd.DataFrame(results)
        df_ml.to_csv(self.results_dir / "model_comparison.csv", index=False)

        # Comparative bar plot (Figure 8)
        plot_classical_comparison(df_ml, self.figures_dir / "figure_08_ml_comparison.png")

        # ROC and PR Curves (Figure 10)
        if roc_dict:
            plot_roc_curves(roc_dict, self.figures_dir / "roc_curves_ml.png")
            plot_pr_curves(pr_dict, self.figures_dir / "figure_10_pr_curves_ml.png")

        # Compare with Paper Table 5
        comp_df = compare_ml_results(results)
        save_comparison_table(comp_df, self.results_dir / "reported_vs_reproduced.csv")
        plot_reported_vs_reproduced(comp_df, self.figures_dir / "figure_16_reported_vs_reproduced.png")

        print("Classical ML experiments complete.")
        return df_ml

    def run_cnn_epoch_experiments(self) -> pd.DataFrame:
        """Execute CNN 10-layer experiments with epoch sizes 14 and 35 (Table 6)."""
        if self.preprocessed is None:
            self.preprocessed = preprocess_data(self.df_raw, test_size=0.20, random_state=self.seed)

        print("Running CNN 10-layer epoch experiments (Table 6)...")
        results = []
        for ep in [14, 35]:
            print(f"  Training CNN 10-Layer for {ep} epochs...")
            res = train_and_evaluate_dl_model(
                architecture="cnn_10_layers",
                X_train=self.preprocessed.X_train,
                y_train=self.preprocessed.y_train,
                X_test=self.preprocessed.X_test,
                y_test=self.preprocessed.y_test,
                epochs=ep,
                batch_size=2048,
                random_state=self.seed,
                dataset_variant="imbalanced",
                verbose=1
            )
            # Plot history for epoch 35 matching Figure 14
            if ep == 35:
                plot_training_history(
                    res["history"],
                    title="Training & Validation History (Epoch size 35)",
                    output_path=self.figures_dir / "figure_14_cnn_epoch_35_history.png"
                )

            clean_res = {k: v for k, v in res.items() if k not in ["y_pred", "y_prob", "model_instance", "history"]}
            clean_res["architecture"] = "cnn_10_layers"
            results.append(clean_res)

        df_epoch = pd.DataFrame(results)
        df_epoch.to_csv(self.results_dir / "cnn_epoch_comparison.csv", index=False)
        print("CNN epoch experiments complete.")
        return df_epoch

    def run_baseline_experiments(self) -> pd.DataFrame:
        """Execute Baseline neural network experiments (Table 7 & Table 10)."""
        if self.preprocessed is None:
            self.preprocessed = preprocess_data(self.df_raw, test_size=0.20, random_state=self.seed)

        print("Running Baseline Neural Network experiments...")
        results = []
        for ep in [14, 20, 35]:
            print(f"  Training Baseline for {ep} epochs...")
            res = train_and_evaluate_dl_model(
                architecture="baseline_nn",
                X_train=self.preprocessed.X_train,
                y_train=self.preprocessed.y_train,
                X_test=self.preprocessed.X_test,
                y_test=self.preprocessed.y_test,
                epochs=ep,
                batch_size=2048,
                random_state=self.seed,
                dataset_variant="imbalanced",
                use_careful_bias=True,
                verbose=1
            )
            clean_res = {k: v for k, v in res.items() if k not in ["y_pred", "y_prob", "model_instance", "history"]}
            clean_res["architecture"] = "baseline_5_layers"
            results.append(clean_res)

        df_base = pd.DataFrame(results)
        df_base.to_csv(self.results_dir / "baseline_experiments.csv", index=False)
        print("Baseline experiments complete.")
        return df_base

    def run_depth_experiments(
        self,
        epochs: int = 100
    ) -> pd.DataFrame:
        """Execute proposed CNN depth experiments (14, 17, 20 layers) matching Table 7 & Figure 17."""
        if self.preprocessed is None:
            self.preprocessed = preprocess_data(self.df_raw, test_size=0.20, random_state=self.seed)

        architectures = [
            ("cnn_14_layers", 14),
            ("cnn_17_layers", 17),
            ("cnn_20_layers", 20)
        ]

        print(f"Running CNN depth experiments ({[a[0] for a in architectures]})...")
        results = []
        depth_rows = []

        for arch_name, layer_count in architectures:
            print(f"  Training {arch_name} ({layer_count} layers) for {epochs} epochs...")
            res = train_and_evaluate_dl_model(
                architecture=arch_name,
                X_train=self.preprocessed.X_train,
                y_train=self.preprocessed.y_train,
                X_test=self.preprocessed.X_test,
                y_test=self.preprocessed.y_test,
                epochs=epochs,
                batch_size=2048,
                random_state=self.seed,
                dataset_variant="imbalanced",
                verbose=1
            )
            clean_res = {k: v for k, v in res.items() if k not in ["y_pred", "y_prob", "model_instance", "history"]}
            clean_res["layers"] = layer_count
            results.append(clean_res)

            depth_rows.append({
                "layers": layer_count,
                "architecture": arch_name,
                "train_accuracy": res["final_train_accuracy_pct"] / 100.0,
                "val_accuracy": res["final_val_accuracy_pct"] / 100.0
            })

        df_depth = pd.DataFrame(results)
        df_depth.to_csv(self.results_dir / "cnn_depth_comparison.csv", index=False)

        plot_depth_comparison(pd.DataFrame(depth_rows), self.figures_dir / "figure_17_cnn_depth_comparison.png")
        print("CNN depth experiments complete.")
        return df_depth

    def run_balanced_data_experiment(
        self,
        epochs: int = 100
    ) -> Dict[str, Any]:
        """Execute balanced data experiment (Section IV-C-9 & Table 7)."""
        print(f"Running balanced dataset experiment for {epochs} epochs...")
        balanced_split = create_presplit_balanced_dataset(
            self.df_raw,
            test_size=0.20,
            random_state=self.seed
        )

        res = train_and_evaluate_dl_model(
            architecture="cnn_14_layers",
            X_train=balanced_split.X_train,
            y_train=balanced_split.y_train,
            X_test=balanced_split.X_test,
            y_test=balanced_split.y_test,
            epochs=epochs,
            batch_size=32,
            random_state=self.seed,
            dataset_variant="balanced",
            verbose=1
        )

        # Plot training history matching Figure 15
        plot_training_history(
            res["history"],
            title="Training & Validation History on Balanced Data (100 Epochs)",
            output_path=self.figures_dir / "figure_15_balanced_cnn_history.png"
        )

        clean_res = {k: v for k, v in res.items() if k not in ["y_pred", "y_prob", "model_instance", "history"]}
        clean_res["sampling_metadata"] = balanced_split.metadata

        df_bal = pd.DataFrame([clean_res])
        df_bal.to_csv(self.results_dir / "balanced_data_results.csv", index=False)
        print("Balanced dataset experiment complete.")
        return clean_res
