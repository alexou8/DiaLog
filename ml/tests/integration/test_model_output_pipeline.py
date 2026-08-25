"""Integration tests for end-to-end model pipeline."""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.models.predictor import GlucosePredictor
from src.utils.output_manager import OutputManager
from src.data.loaders import prepare_features, split_train_test


@pytest.mark.integration
class TestModelOutputPipeline:
    """Integration tests for complete model training and output pipeline."""
    
    def test_full_training_pipeline(self, sample_glucose_data, temp_model_dir, temp_output_dir):
        """Test complete pipeline: data -> train -> predict -> save."""
        # Prepare data
        feature_cols = ['carbs', 'insulin', 'activity_minutes']
        target_col = 'glucose'
        
        X, y = prepare_features(sample_glucose_data, feature_cols, target_col)
        X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.2)
        
        # Train model
        config = {'random_state': 42, 'n_estimators': 50}
        model = GlucosePredictor(config)
        model.train(X_train, y_train)
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Save model
        model_path = temp_model_dir / "glucose_model.pkl"
        model.save(model_path)
        assert model_path.exists()
        
        # Save predictions
        output_manager = OutputManager(temp_output_dir)
        pred_df = pd.DataFrame({
            'predicted_glucose': predictions,
            'actual_glucose': y_test
        })
        
        metadata = {
            'features': feature_cols,
            'n_samples': len(X_test)
        }
        
        pred_path = output_manager.save_predictions(
            pred_df,
            model_name="glucose_predictor",
            metadata=metadata
        )
        
        assert pred_path.exists()
        
        # Verify we can load predictions
        loaded_pred = output_manager.load_predictions(pred_path)
        assert len(loaded_pred) == len(predictions)
    
    def test_model_persistence_pipeline(self, sample_training_data, temp_model_dir):
        """Test that saved model produces same predictions after loading."""
        X, y = sample_training_data
        X_train, X_test = X[:80], X[80:]
        y_train = y[:80]
        
        # Train and save model
        config = {'random_state': 42}
        model = GlucosePredictor(config)
        model.train(X_train, y_train)
        
        original_predictions = model.predict(X_test)
        
        model_path = temp_model_dir / "test_model.pkl"
        model.save(model_path)
        
        # Load model and predict
        new_model = GlucosePredictor(config)
        new_model.load(model_path)
        loaded_predictions = new_model.predict(X_test)
        
        # Predictions should be identical
        np.testing.assert_array_almost_equal(original_predictions, loaded_predictions)
    
    def test_evaluate_and_save_metrics(self, sample_training_data, temp_output_dir):
        """Test evaluation and metrics saving pipeline."""
        X, y = sample_training_data
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]
        
        # Train model
        config = {'random_state': 42}
        model = GlucosePredictor(config)
        model.train(X_train, y_train)
        
        # Evaluate
        metrics = model.evaluate(X_test, y_test)
        
        # Save metrics
        output_manager = OutputManager(temp_output_dir)
        metrics_path = output_manager.save_metrics(
            metrics,
            model_name="glucose_predictor",
            run_id="test_run"
        )
        
        assert metrics_path.exists()
        
        # Verify metrics are valid
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert all(isinstance(v, float) for v in metrics.values())
    
    def test_reload_and_continue_prediction(self, sample_glucose_data, temp_model_dir, temp_output_dir):
        """Test loading saved model and making new predictions."""
        # Prepare and split data
        feature_cols = ['carbs', 'insulin', 'activity_minutes']
        X, y = prepare_features(sample_glucose_data, feature_cols, 'glucose')
        X_train, X_test, y_train, y_test = split_train_test(X, y)
        
        # Train and save model (simulation of first session)
        config = {'random_state': 42}
        model = GlucosePredictor(config)
        model.train(X_train, y_train)
        
        model_path = temp_model_dir / "saved_model.pkl"
        model.save(model_path)
        
        # Load model and make predictions (simulation of second session)
        new_model = GlucosePredictor(config)
        new_model.load(model_path)
        
        predictions = new_model.predict(X_test)
        
        # Save predictions
        output_manager = OutputManager(temp_output_dir)
        pred_df = pd.DataFrame({
            'predicted': predictions,
            'actual': y_test
        })
        
        pred_path = output_manager.save_predictions(pred_df, "glucose_predictor")
        
        # Verify retrieval
        latest = output_manager.get_latest_predictions("glucose_predictor")
        assert latest is not None
        assert len(latest) == len(predictions)
    
    def test_multiple_models_workflow(self, sample_training_data, temp_model_dir, temp_output_dir):
        """Test workflow with multiple model versions."""
        X, y = sample_training_data
        
        # Train multiple model configurations
        configs = [
            {'n_estimators': 50, 'max_depth': 5, 'random_state': 42},
            {'n_estimators': 100, 'max_depth': 10, 'random_state': 42},
        ]
        
        output_manager = OutputManager(temp_output_dir)
        
        for i, config in enumerate(configs):
            model = GlucosePredictor(config)
            model.train(X, y)
            
            # Save model
            model_path = temp_model_dir / f"model_v{i}.pkl"
            model.save(model_path)
            
            # Evaluate and save metrics
            metrics = model.evaluate(X, y)
            output_manager.save_metrics(metrics, f"model_v{i}", f"run_{i}")
        
        # Verify all models were saved
        assert (temp_model_dir / "model_v0.pkl").exists()
        assert (temp_model_dir / "model_v1.pkl").exists()
