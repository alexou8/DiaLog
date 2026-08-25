"""Unit tests for model implementations."""
import pytest
import numpy as np
from pathlib import Path
from src.models.predictor import GlucosePredictor


@pytest.mark.unit
class TestGlucosePredictor:
    """Test cases for GlucosePredictor model."""
    
    def test_model_initialization(self):
        """Test that model initializes with correct config."""
        config = {
            'n_estimators': 50,
            'random_state': 42,
            'max_depth': 5
        }
        model = GlucosePredictor(config)
        
        assert model.config == config
        assert model.model is not None
        assert model.is_trained is False
        assert model.model.n_estimators == 50
        assert model.model.max_depth == 5
    
    def test_model_initialization_defaults(self):
        """Test that model uses defaults when config is empty."""
        config = {}
        model = GlucosePredictor(config)
        
        assert model.model.n_estimators == 100
        assert model.model.max_depth == 10
        assert model.model.random_state == 42
    
    def test_model_training(self, sample_training_data):
        """Test that model can be trained."""
        X, y = sample_training_data
        config = {'random_state': 42}
        model = GlucosePredictor(config)
        
        assert model.is_trained is False
        model.train(X, y)
        assert model.is_trained is True
    
    def test_prediction_shape(self, sample_training_data):
        """Test that predictions have correct shape."""
        X, y = sample_training_data
        config = {'random_state': 42}
        model = GlucosePredictor(config)
        
        model.train(X, y)
        predictions = model.predict(X)
        
        assert predictions.shape == y.shape
        assert len(predictions) == len(X)
    
    def test_prediction_without_training(self, sample_training_data):
        """Test that prediction without training raises error."""
        X, _ = sample_training_data
        config = {'random_state': 42}
        model = GlucosePredictor(config)
        
        with pytest.raises(ValueError, match="Model not trained"):
            model.predict(X)
    
    def test_model_save_load(self, sample_training_data, temp_model_dir):
        """Test that model can be saved and loaded."""
        X, y = sample_training_data
        config = {'random_state': 42}
        model = GlucosePredictor(config)
        
        # Train and save
        model.train(X, y)
        model_path = temp_model_dir / "test_model.pkl"
        model.save(model_path)
        
        assert model_path.exists()
        
        # Load and compare predictions
        original_predictions = model.predict(X[:10])
        
        model2 = GlucosePredictor(config)
        model2.load(model_path)
        loaded_predictions = model2.predict(X[:10])
        
        assert model2.is_trained is True
        np.testing.assert_array_almost_equal(original_predictions, loaded_predictions)
    
    def test_evaluate_metrics(self, sample_training_data):
        """Test that evaluation returns correct metrics."""
        X, y = sample_training_data
        config = {'random_state': 42}
        model = GlucosePredictor(config)
        
        model.train(X, y)
        metrics = model.evaluate(X, y)
        
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        
        assert metrics['rmse'] >= 0
        assert metrics['mae'] >= 0
        assert metrics['r2'] <= 1.0
    
    @pytest.mark.parametrize("n_features", [5, 10, 20])
    def test_different_feature_counts(self, random_seed, n_features):
        """Test model with different numbers of features."""
        X = np.random.randn(100, n_features)
        y = np.random.randn(100) * 20 + 130
        
        config = {'random_state': 42}
        model = GlucosePredictor(config)
        
        model.train(X, y)
        predictions = model.predict(X)
        
        assert predictions.shape == y.shape
    
    @pytest.mark.parametrize("n_estimators", [10, 50, 100])
    def test_different_n_estimators(self, sample_training_data, n_estimators):
        """Test model with different n_estimators values."""
        X, y = sample_training_data
        config = {'n_estimators': n_estimators, 'random_state': 42}
        model = GlucosePredictor(config)
        
        model.train(X, y)
        assert model.model.n_estimators == n_estimators
