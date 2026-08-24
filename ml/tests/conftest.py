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


@pytest.fixture
def synthetic_event_rows() -> list[tuple]:
    """Synthetic (SYNTHETIC -- no clinical validity) raw event rows in the
    schema expected by ``src.features.to_dataframe``: (timestamp,
    event_type, carbs_g, med_name, med_units, glucose_mgdl, notes).

    Glucose is deliberately correlated with recent carbs so that a model
    trained on this fixture has genuine (if fixture-scale) signal to find,
    which lets tests distinguish "the pipeline is wired correctly" from
    "the model happens to score well by chance".
    """
    rng = np.random.default_rng(123)
    start = pd.Timestamp("2026-01-01")
    rows: list[tuple] = []
    for day in range(40):
        day_ts = start + pd.Timedelta(days=day)
        rows.append((day_ts + pd.Timedelta(hours=8), "med", None, "Metformin", 1.0, None, "am dose"))
        breakfast_carbs = float(rng.uniform(20, 80))
        lunch_carbs = float(rng.uniform(20, 90))
        rows.append((day_ts + pd.Timedelta(hours=8, minutes=10), "meal", breakfast_carbs, None, None, None, None))
        rows.append((day_ts + pd.Timedelta(hours=12), "meal", lunch_carbs, None, None, None, None))
        for meal_hour, carbs in [(8, breakfast_carbs), (12, lunch_carbs)]:
            glucose = float(np.clip(110 + 1.1 * carbs + rng.normal(0, 15), 70, 320))
            ts = day_ts + pd.Timedelta(hours=meal_hour + 2)
            rows.append((ts, "glucose", None, None, None, glucose, "post-meal"))
    return rows


@pytest.fixture
def synthetic_events_df(synthetic_event_rows):
    """The synthetic_event_rows fixture converted to the sorted events dataframe."""
    from src.features import to_dataframe

    return to_dataframe(synthetic_event_rows)


@pytest.fixture
def synthetic_examples_df(synthetic_events_df):
    """Feature-engineered training examples built from the synthetic events fixture."""
    from src.features import FeatureConfig, build_training_examples

    return build_training_examples(synthetic_events_df, config=FeatureConfig())
