"""Data validation utilities for DiaLog."""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def validate_glucose_data(df: pd.DataFrame, required_columns: Optional[List[str]] = None) -> Dict[str, any]:
    """
    Validate glucose monitoring data.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
    
    Returns:
        Dictionary with validation results and statistics
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    # Check required columns
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            results['valid'] = False
            results['errors'].append(f"Missing required columns: {missing_cols}")
    
    # Check for empty DataFrame
    if len(df) == 0:
        results['valid'] = False
        results['errors'].append("DataFrame is empty")
        return results
    
    # Check for missing values
    missing_counts = df.isnull().sum()
    if missing_counts.any():
        results['warnings'].append(f"Missing values found: {missing_counts[missing_counts > 0].to_dict()}")
    
    # Validate glucose values (if column exists)
    if 'glucose' in df.columns or 'glucose_mg_dl' in df.columns:
        glucose_col = 'glucose' if 'glucose' in df.columns else 'glucose_mg_dl'
        glucose_vals = df[glucose_col].dropna()
        
        if len(glucose_vals) > 0:
            # Check range (typical glucose range: 40-400 mg/dL)
            if (glucose_vals < 40).any() or (glucose_vals > 400).any():
                results['warnings'].append("Glucose values outside typical range (40-400 mg/dL)")
            
            results['stats']['glucose_mean'] = float(glucose_vals.mean())
            results['stats']['glucose_std'] = float(glucose_vals.std())
            results['stats']['glucose_min'] = float(glucose_vals.min())
            results['stats']['glucose_max'] = float(glucose_vals.max())
    
    # General statistics
    results['stats']['n_rows'] = len(df)
    results['stats']['n_columns'] = len(df.columns)
    
    if results['valid']:
        logger.info("Data validation passed")
    else:
        logger.error(f"Data validation failed: {results['errors']}")
    
    return results


def check_feature_quality(X: np.ndarray, feature_names: Optional[List[str]] = None) -> Dict[str, any]:
    """
    Check quality of feature matrix.
    
    Args:
        X: Feature matrix
        feature_names: Optional list of feature names
    
    Returns:
        Dictionary with quality metrics
    """
    results = {
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'has_nan': bool(np.isnan(X).any()),
        'has_inf': bool(np.isinf(X).any()),
    }
    
    if feature_names and len(feature_names) != X.shape[1]:
        logger.warning(f"Feature name count mismatch: {len(feature_names)} names, {X.shape[1]} features")
    
    # Check for constant features
    feature_stds = np.std(X, axis=0)
    constant_features = np.where(feature_stds == 0)[0]
    if len(constant_features) > 0:
        results['constant_features'] = constant_features.tolist()
        logger.warning(f"Found {len(constant_features)} constant features")
    
    return results
