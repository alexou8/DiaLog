"""Shared pytest fixtures for DiaLog tests."""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    return 42


@pytest.fixture
def sample_glucose_data():
    """Generate sample glucose monitoring data."""
    dates = pd.date_range('2026-01-01', periods=100, freq='1h')
    data = {
        'timestamp': dates,
        'glucose': np.random.normal(130, 20, 100),
        'carbs': np.random.uniform(0, 60, 100),
        'insulin': np.random.uniform(0, 10, 100),
        'activity_minutes': np.random.uniform(0, 60, 100)
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_training_data(random_seed):
    """Generate sample training data (X, y)."""
    X = np.random.randn(100, 10)
    y = np.random.randn(100) * 20 + 130  # Glucose-like values
    return X, y


@pytest.fixture
def temp_model_dir(tmp_path):
    """Create temporary model directory."""
    model_dir = tmp_path / "models"
    model_dir.mkdir(exist_ok=True)
    return model_dir


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(exist_ok=True)
    return output_dir
