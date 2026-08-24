"""Property-based tests for models using Hypothesis."""
import pytest
import numpy as np
from hypothesis import given, strategies as st, settings
from src.models.predictor import GlucosePredictor


@pytest.mark.unit
@pytest.mark.slow
class TestModelProperties:
    """Property-based tests for model behavior."""
    
    @given(
        n_samples=st.integers(min_value=10, max_value=100),
        n_features=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=10, deadline=None)
    def test_prediction_shape_property(self, n_samples, n_features):
        """Property: predictions should always match input shape."""
        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples) * 20 + 130
        
        config = {'random_state': 42}
        model = GlucosePredictor(config)
        model.train(X, y)
        
        predictions = model.predict(X)
        assert predictions.shape == (n_samples,)
    
    @given(
        n_estimators=st.integers(min_value=10, max_value=200),
        max_depth=st.integers(min_value=2, max_value=20)
    )
    @settings(max_examples=10, deadline=None)
    def test_model_config_property(self, n_estimators, max_depth):
        """Property: model should respect configuration parameters."""
        config = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'random_state': 42
        }
        model = GlucosePredictor(config)
        
        assert model.model.n_estimators == n_estimators
        assert model.model.max_depth == max_depth
    
    def test_deterministic_predictions(self, sample_training_data):
        """Property: same random seed should give same predictions."""
        X, y = sample_training_data
        
        # Train two models with same random seed
        model1 = GlucosePredictor({'random_state': 42})
        model1.train(X, y)
        pred1 = model1.predict(X)
        
        model2 = GlucosePredictor({'random_state': 42})
        model2.train(X, y)
        pred2 = model2.predict(X)
        
        np.testing.assert_array_almost_equal(pred1, pred2)
    
    def test_metrics_bounds(self, sample_training_data):
        """Property: metrics should be within expected bounds."""
        X, y = sample_training_data
        config = {'random_state': 42}
        model = GlucosePredictor(config)
        
        model.train(X, y)
        metrics = model.evaluate(X, y)
        
        # RMSE and MAE should be non-negative
        assert metrics['rmse'] >= 0
        assert metrics['mae'] >= 0
        
        # R2 should be <= 1.0
        assert metrics['r2'] <= 1.0
    
    def test_predictions_are_numeric(self, sample_training_data):
        """Property: predictions should be finite numeric values."""
        X, y = sample_training_data
        config = {'random_state': 42}
        model = GlucosePredictor(config)
        
        model.train(X, y)
        predictions = model.predict(X)
        
        assert np.all(np.isfinite(predictions))
        assert predictions.dtype in [np.float64, np.float32]
