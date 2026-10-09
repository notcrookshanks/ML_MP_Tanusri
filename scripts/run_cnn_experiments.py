#!/usr/bin/env python3
"""CLI script to run Deep Learning & CNN experiments."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.experiment_runner import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(description="Run Deep Learning & CNN Experiments.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to base configuration YAML"
    )
    parser.add_argument(
        "--depth-epochs",
        type=int,
        default=10,
        help="Epochs for 14/17/20-layer depth experiments (default 10 for interactive run; paper reports up to 100)"
    )
    parser.add_argument(
        "--skip-depth",
        action="store_true",
        help="Skip depth experiments"
    )

    args = parser.parse_args()

    runner = ExperimentRunner(config_path=args.config)
    runner.load_and_validate()

    print("\n--- Running Baseline Model Experiments ---")
    df_base = runner.run_baseline_experiments()
    print(df_base[["model", "epochs", "accuracy_pct", "precision_pct", "recall_pct"]].to_string(index=False))

    print("\n--- Running CNN 10-Layer Epoch Experiments (Table 6) ---")
    df_epoch = runner.run_cnn_epoch_experiments()
    print(df_epoch[["model", "epochs", "accuracy_pct", "precision_pct", "recall_pct"]].to_string(index=False))

    print("\n--- Running Balanced Dataset Experiment (Section IV-C-9) ---")
    res_bal = runner.run_balanced_data_experiment(epochs=min(args.depth_epochs * 2, 50))
    print(f"Balanced CNN Val Accuracy: {res_bal['final_val_accuracy_pct']:.2f}%, F1: {res_bal['f1_score_pct']:.2f}%")

    if not args.skip_depth:
        print(f"\n--- Running CNN Depth Experiments ({args.depth_epochs} Epochs) ---")
        df_depth = runner.run_depth_experiments(epochs=args.depth_epochs)
        print(df_depth[["model", "layers", "final_train_accuracy_pct", "final_val_accuracy_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
