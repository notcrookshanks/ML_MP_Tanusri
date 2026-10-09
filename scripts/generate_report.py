#!/usr/bin/env python3
"""CLI script to generate reproduction report from existing result artifacts."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.reporting.generate_report import generate_markdown_report


def main():
    parser = argparse.ArgumentParser(description="Generate markdown reproduction report.")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="reports/results",
        help="Directory containing results CSVs"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/reproduction_report.md",
        help="Output markdown file path"
    )

    args = parser.parse_args()
    report = generate_markdown_report(results_dir=args.results_dir, output_path=args.output)
    print(f"Reproduction report generated at: {args.output}")


if __name__ == "__main__":
    main()
