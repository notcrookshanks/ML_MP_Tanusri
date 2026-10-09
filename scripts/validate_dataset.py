#!/usr/bin/env python3
"""CLI script to validate the credit card fraud dataset."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.load_data import load_raw_data
from src.data.validate_data import validate_dataset, save_validation_report, DataValidationError


def main():
    parser = argparse.ArgumentParser(description="Validate Credit Card Fraud Dataset integrity.")
    parser.add_argument(
        "--path",
        type=str,
        default="data/raw/creditcard.csv",
        help="Path to creditcard.csv file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/results/data_validation_report.json",
        help="Output JSON report path"
    )
    parser.add_argument(
        "--non-strict",
        action="store_true",
        help="Disable strict row count check (for synthetic testing)"
    )

    args = parser.parse_args()

    print(f"Loading dataset from: {args.path}")
    try:
        df = load_raw_data(args.path)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print("Running validation checks...")
    try:
        report = validate_dataset(df, strict_row_count=not args.non_strict)
        save_validation_report(report, args.output)
        print("Dataset validation PASSED!")
        print(f"Validation report saved to: {args.output}")
        print(f"Total Rows: {report['statistics']['shape'][0]:,}")
        print(f"Legitimate Transactions: {report['statistics']['class_distribution']['legitimate']:,}")
        print(f"Fraudulent Transactions: {report['statistics']['class_distribution']['fraudulent']:,} ({report['statistics']['class_distribution']['fraud_percentage']}%)")
    except DataValidationError as e:
        print(f"Validation FAILED: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
