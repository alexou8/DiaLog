"""Example: Make predictions with a trained model."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.models.predictor import GlucosePredictor
from src.utils.config import Config
from src.utils.output_manager import OutputManager


def main():
    print("🔮 DiaLog Prediction Example\n")
    
    # Setup
    config_obj = Config()
    config = {
        'random_state': 42,
        'model_dir': config_obj.MODELS_DIR
    }
    
    # Load model
    model_path = config_obj.MODELS_DIR / "glucose_predictor_v1.pkl"
    if not model_path.exists():
        print("⚠️  Model not found. Run examples/train_model_example.py first.")
        return
    
    model = GlucosePredictor(config)
    model.load(model_path)
    print(f"✅ Model loaded from {model_path}\n")
    
    # Load test data
    data_path = config_obj.DATA_DIR / "sample_glucose_data.csv"
    df = pd.read_csv(data_path)
    
    # Use last 20% as new data
    split_idx = int(0.8 * len(df))
    df_new = df[split_idx:].reset_index(drop=True)
    
    # Prepare features
    feature_cols = ['carbs_grams', 'insulin_units', 'activity_minutes', 
                   'stress_level', 'sleep_hours']
    X_new = df_new[feature_cols].values
    
    # Make predictions
    print(f"🎯 Making predictions on {len(X_new)} samples...")
    predictions = model.predict(X_new)
    
    # Create predictions dataframe
    pred_df = pd.DataFrame({
        'timestamp': df_new['timestamp'],
        'actual_glucose': df_new['glucose_mg_dl'],
        'predicted_glucose': predictions,
        'error': df_new['glucose_mg_dl'] - predictions
    })
    
    print("✅ Predictions completed!\n")
    print("Sample predictions:")
    print(pred_df.head(10))
    print(f"\nMean absolute error: {pred_df['error'].abs().mean():.2f} mg/dL\n")
    
    # Save predictions
    output_manager = OutputManager(config_obj.OUTPUTS_DIR)
    metadata = {
        'model_version': 'v1',
        'features_used': feature_cols,
        'n_predictions': len(predictions)
    }
    
    output_path = output_manager.save_predictions(
        pred_df,
        model_name="glucose_predictor",
        metadata=metadata
    )
    print(f"💾 Predictions saved to {output_path}")


if __name__ == "__main__":
    main()
