"""Data loading utilities for the Credit Card Fraud dataset."""

import os
from pathlib import Path
from typing import Optional, Union
import pandas as pd


def load_raw_data(
    filepath: Union[str, Path] = "data/raw/creditcard.csv",
    nrows: Optional[int] = None
) -> pd.DataFrame:
    """Load the raw credit card dataset from a CSV file.

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file.
    nrows : int, optional
        Number of rows to load (useful for fast smoke tests or debugging).

    Returns
    -------
    pd.DataFrame
        Loaded dataset dataframe.

    Raises
    ------
    FileNotFoundError
        If the dataset file does not exist.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Please ensure creditcard.csv "
            "is located in data/raw/ or provide a valid path."
        )

    df = pd.read_csv(path, nrows=nrows)
    return df
