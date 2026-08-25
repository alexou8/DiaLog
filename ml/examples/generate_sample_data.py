"""Generate sample diabetes tracking data for demonstration."""
import numpy as np
import pandas as pd
from pathlib import Path


def generate_glucose_data(n_days: int = 30, samples_per_day: int = 24) -> pd.DataFrame:
    """Generate realistic glucose monitoring data."""
    np.random.seed(42)
    
    n_samples = n_days * samples_per_day
    dates = pd.date_range('2026-01-01', periods=n_samples, freq='1h')
    
    # Simulate realistic glucose patterns
    time_of_day = np.array([d.hour for d in dates])
    
    # Base glucose with daily pattern
    base_glucose = 120 + 20 * np.sin(2 * np.pi * time_of_day / 24)
    
    # Add meal spikes (breakfast, lunch, dinner)
    meal_times = [7, 12, 18]
    meal_effect = np.zeros(n_samples)
    for meal_time in meal_times:
        meal_mask = np.abs(time_of_day - meal_time) < 2
        meal_effect[meal_mask] = 30 * np.exp(-0.5 * ((time_of_day[meal_mask] - meal_time) ** 2))
    
    glucose = base_glucose + meal_effect + np.random.normal(0, 10, n_samples)
    glucose = np.clip(glucose, 70, 250)  # Realistic range
    
    data = {
        'timestamp': dates,
        'glucose_mg_dl': glucose,
        'carbs_grams': np.random.uniform(0, 60, n_samples),
        'insulin_units': np.random.uniform(0, 10, n_samples),
        'activity_minutes': np.random.uniform(0, 60, n_samples),
        'stress_level': np.random.randint(1, 6, n_samples),
        'sleep_hours': np.random.uniform(6, 9, n_samples)
    }
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    # Generate data
    df = generate_glucose_data(n_days=30)
    
    # Save to data directory
    output_path = Path(__file__).parent.parent / "data" / "sample_glucose_data.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"✅ Generated {len(df)} samples")
    print(f"✅ Saved to {output_path}")
    print(f"\nData preview:")
    print(df.head(10))
    print(f"\nData statistics:")
    print(df.describe())
