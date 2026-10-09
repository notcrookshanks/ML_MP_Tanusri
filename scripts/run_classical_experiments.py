#!/usr/bin/env python3
"""CLI script to run classical ML experiments (Table 5)."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.experiment_runner import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(description="Run Classical Machine Learning Experiments.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to base configuration YAML"
    )
    parser.add_argument(
        "--svm-samples",
        type=int,
        default=30000,
        help="Subsample limit for SVM training (default 30,000 for tractability)"
    )

    args = parser.parse_args()

    runner = ExperimentRunner(config_path=args.config)
    runner.load_and_validate()
    runner.generate_eda_figures()
    runner.run_feature_ranking()
    df_ml = runner.run_classical_ml_experiments(svm_max_samples=args.svm_samples)

    print("\n=== Classical ML Benchmark Results (Table 5) ===")
    print(df_ml[["model", "accuracy_pct", "f1_score_pct", "precision_pct", "recall_pct", "total_runtime_sec"]].to_string(index=False))


if __name__ == "__main__":
    main()
