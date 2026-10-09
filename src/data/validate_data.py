"""Data validation module ensuring dataset integrity and conformity to paper specifications."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

EXPECTED_COLUMNS: List[str] = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount",
    "Class"
]

EXPECTED_TOTAL_ROWS: int = 284807
EXPECTED_FRAUD_COUNT: int = 492
EXPECTED_LEGIT_COUNT: int = 284315


class DataValidationError(ValueError):
    """Exception raised when dataset fails integrity validation."""
    pass


def validate_dataset(
    df: pd.DataFrame,
    strict_row_count: bool = True
) -> Dict[str, Any]:
    """Validate credit card dataframe against expected schema and characteristics.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to validate.
    strict_row_count : bool, default=True
        Whether to strictly enforce the exact 284,807 row count.
        Set to False for synthetic smoke testing.

    Returns
    -------
    dict
        Structured validation report.

    Raises
    ------
    DataValidationError
        If schema, data types, nulls, or class counts fail validation.
    """
    report: Dict[str, Any] = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "statistics": {}
    }

    # 1. Column Schema Check
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in EXPECTED_COLUMNS]

    if missing_cols:
        err = f"Missing required columns: {missing_cols}"
        report["errors"].append(err)
        report["is_valid"] = False

    if extra_cols:
        report["warnings"].append(f"Unexpected extra columns: {extra_cols}")

    if not report["is_valid"]:
        raise DataValidationError("; ".join(report["errors"]))

    # 2. Row Count & Shape
    total_rows = len(df)
    total_cols = len(df.columns)
    report["statistics"]["shape"] = (total_rows, total_cols)

    if strict_row_count and total_rows != EXPECTED_TOTAL_ROWS:
        err = (
            f"Row count mismatch: expected {EXPECTED_TOTAL_ROWS}, "
            f"found {total_rows}."
        )
        report["errors"].append(err)
        report["is_valid"] = False

    # 3. Missing (Null) Values Check
    null_counts = df.isnull().sum().to_dict()
    total_nulls = int(df.isnull().sum().sum())
    report["statistics"]["null_counts"] = null_counts
    report["statistics"]["total_nulls"] = total_nulls

    if total_nulls > 0:
        err = f"Dataset contains {total_nulls} null/missing values."
        report["errors"].append(err)
        report["is_valid"] = False

    # 4. Data Types Check
    dtypes_check = {}
    for col in df.columns:
        if col == "Class":
            if not np.issubdtype(df[col].dtype, np.integer):
                err = f"Column 'Class' must be integer type, found {df[col].dtype}."
                report["errors"].append(err)
                report["is_valid"] = False
        else:
            if not np.issubdtype(df[col].dtype, np.floating) and not np.issubdtype(df[col].dtype, np.integer):
                err = f"Column '{col}' must be numeric, found {df[col].dtype}."
                report["errors"].append(err)
                report["is_valid"] = False
        dtypes_check[col] = str(df[col].dtype)
    report["statistics"]["dtypes"] = dtypes_check

    # 5. Target Class Distribution Check
    class_counts = df["Class"].value_counts().to_dict()
    fraud_count = int(class_counts.get(1, 0))
    legit_count = int(class_counts.get(0, 0))
    report["statistics"]["class_distribution"] = {
        "legitimate": legit_count,
        "fraudulent": fraud_count,
        "fraud_percentage": round(fraud_count / max(total_rows, 1) * 100, 4)
    }

    unique_classes = set(df["Class"].unique())
    if not unique_classes.issubset({0, 1}):
        err = f"Invalid target classes found: {unique_classes}. Expected only {{0, 1}}."
        report["errors"].append(err)
        report["is_valid"] = False

    if strict_row_count:
        if fraud_count != EXPECTED_FRAUD_COUNT:
            err = (
                f"Fraud count mismatch: expected {EXPECTED_FRAUD_COUNT}, "
                f"found {fraud_count}."
            )
            report["errors"].append(err)
            report["is_valid"] = False

        if legit_count != EXPECTED_LEGIT_COUNT:
            err = (
                f"Legitimate count mismatch: expected {EXPECTED_LEGIT_COUNT}, "
                f"found {legit_count}."
            )
            report["errors"].append(err)
            report["is_valid"] = False

    # 6. Duplicates Check
    duplicate_count = int(df.duplicated().sum())
    report["statistics"]["duplicate_rows"] = duplicate_count
    if duplicate_count > 0:
        report["warnings"].append(
            f"Dataset contains {duplicate_count} duplicated rows (standard in 2013 benchmark)."
        )

    # 7. Numerical Range Check for Amount and Time
    report["statistics"]["amount_summary"] = {
        "min": float(df["Amount"].min()),
        "max": float(df["Amount"].max()),
        "mean": float(df["Amount"].mean()),
        "std": float(df["Amount"].std())
    }
    report["statistics"]["time_summary"] = {
        "min": float(df["Time"].min()),
        "max": float(df["Time"].max()),
        "duration_hours": round(float(df["Time"].max() - df["Time"].min()) / 3600.0, 2)
    }

    if not report["is_valid"]:
        raise DataValidationError("; ".join(report["errors"]))

    return report


def save_validation_report(
    report: Dict[str, Any],
    output_path: Union[str, Path] = "reports/results/data_validation_report.json"
) -> None:
    """Save the validation report to disk as JSON."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
