"""Data loading utilities for DiaLog."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


def load_glucose_data(filepath: Path, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load glucose monitoring data from CSV.
    
    Args:
        filepath: Path to CSV file
        columns: Optional list of columns to load
    
    Returns:
        DataFrame with glucose data
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath, usecols=columns)
    logger.info(f"Loaded {len(df)} rows")
    
    return df


def prepare_features(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str
) -> tuple:
    """
    Prepare feature matrix X and target vector y from DataFrame.
    
    Args:
        df: Input DataFrame
        feature_columns: List of feature column names
        target_column: Name of target column
    
    Returns:
        Tuple of (X, y) as numpy arrays
    """
    missing_cols = set(feature_columns + [target_column]) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in DataFrame: {missing_cols}")
    
    X = df[feature_columns].values
    y = df[target_column].values
    
    logger.info(f"Prepared features: X shape {X.shape}, y shape {y.shape}")
    
    return X, y


def split_train_test(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    shuffle: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and testing sets.

    Defaults to a CHRONOLOGICAL split (shuffle=False): the last
    ``test_size`` fraction of rows, in the order given, becomes the test
    set. Callers passing pre-time-sorted arrays (as this pipeline does) get
    a leakage-free split by default. Pass shuffle=True only for genuinely
    i.i.d. data where row order carries no temporal meaning.

    Args:
        X: Feature matrix
        y: Target vector
        test_size: Fraction of data to use for testing
        random_state: Random seed (only used when shuffle=True)
        shuffle: If True, use a random shuffled split (sklearn
            train_test_split). If False (default), split chronologically by
            taking the trailing ``test_size`` fraction of rows as test data.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    if shuffle:
        from sklearn.model_selection import train_test_split

        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
        )

    n = len(X)
    split_idx = int(round(n * (1 - test_size)))
    split_idx = max(1, min(split_idx, n - 1)) if n > 1 else n
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    return X_train, X_test, y_train, y_test
