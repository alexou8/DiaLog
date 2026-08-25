"""Example: Train a glucose prediction model."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from src.models.predictor import GlucosePredictor
from src.utils.config import Config
from src.utils.output_manager import OutputManager


def main():
    print("🏥 DiaLog Model Training Example\n")
    
    # Setup
    config_obj = Config()
    config = {
        'n_estimators': 100,
        'random_state': 42,
        'max_depth': 10,
        'model_dir': config_obj.MODELS_DIR
    }
    
    # Load sample data
    data_path = config_obj.DATA_DIR / "sample_glucose_data.csv"
    if not data_path.exists():
        print("⚠️  Sample data not found. Run examples/generate_sample_data.py first.")
        return
    
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df)} samples\n")
    
    # Prepare features and target
    feature_cols = ['carbs_grams', 'insulin_units', 'activity_minutes', 
                   'stress_level', 'sleep_hours']
    X = df[feature_cols].values
    y = df['glucose_mg_dl'].values
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}\n")
    
    # Train model
    print("🚀 Training model...")
    model = GlucosePredictor(config)
    model.train(X_train, y_train)
    print("✅ Model trained!\n")
    
    # Evaluate
    print("📊 Evaluating model...")
    metrics = model.evaluate(X_test, y_test)
    print(f"RMSE: {metrics['rmse']:.2f} mg/dL")
    print(f"MAE: {metrics['mae']:.2f} mg/dL")
    print(f"R²: {metrics['r2']:.3f}\n")
    
    # Save model
    model_path = config_obj.MODELS_DIR / "glucose_predictor_v1.pkl"
    model.save(model_path)
    print(f"💾 Model saved to {model_path}\n")
    
    # Save metrics
    output_manager = OutputManager(config_obj.OUTPUTS_DIR)
    metrics_path = output_manager.save_metrics(metrics, "glucose_predictor", "v1")
    print(f"📈 Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
