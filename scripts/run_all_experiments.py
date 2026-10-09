#!/usr/bin/env python3
"""CLI script to run all reproduction experiments end-to-end and generate the report."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.experiment_runner import ExperimentRunner
from src.reporting.generate_report import generate_markdown_report


def main():
    parser = argparse.ArgumentParser(description="Execute full research reproduction pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to base configuration YAML"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run in accelerated mode with fewer epochs for rapid verification"
    )

    args = parser.parse_args()

    print("=================================================================")
    print("STARTING COMPLETE RESEARCH REPRODUCTION PIPELINE")
    print("Paper: Credit Card Fraud Detection Using State-of-the-Art ML and DL Algorithms")
    print("=================================================================\n")

    runner = ExperimentRunner(config_path=args.config)

    # 1. Load and Validate
    runner.load_and_validate()

    # 2. EDA Figures
    runner.generate_eda_figures()

    # 3. Feature Selection
    runner.run_feature_ranking()

    # 4. Classical ML Experiments
    runner.run_classical_ml_experiments()

    # 5. Baseline DL Experiments
    runner.run_baseline_experiments()

    # 6. CNN 10-Layer Epoch Experiments (Table 6)
    runner.run_cnn_epoch_experiments()

    # 7. Balanced Dataset Experiment (Section IV-C-9)
    bal_epochs = 20 if args.quick else 50
    runner.run_balanced_data_experiment(epochs=bal_epochs)

    # 8. CNN Depth Experiments (14, 17, 20 Layers)
    depth_epochs = 10 if args.quick else 25
    runner.run_depth_experiments(epochs=depth_epochs)

    # 9. Generate Markdown Reproduction Report
    print("\nGenerating comprehensive reproduction report...")
    report_text = generate_markdown_report()
    print("Reproduction report generated successfully: reports/reproduction_report.md")

    print("\n=================================================================")
    print("ALL REPRODUCTION EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=================================================================")


if __name__ == "__main__":
    main()
