"""Glucose prediction model implementation."""
import numpy as np
import pickle
from pathlib import Path
from typing import Dict
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from .base import BaseModel


class GlucosePredictor(BaseModel):
    """Random Forest model for glucose level prediction."""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.model = RandomForestRegressor(
            n_estimators=config.get('n_estimators', 100),
            random_state=config.get('random_state', 42),
            max_depth=config.get('max_depth', 10)
        )
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the Random Forest model."""
        self.model.fit(X, y)
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make glucose level predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)
    
    def save(self, path: Path) -> None:
        """Save model to disk using pickle."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load(self, path: Path) -> None:
        """Load model from disk."""
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        self.is_trained = True
    
    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute regression metrics."""
        return {
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'r2': float(r2_score(y_true, y_pred))
        }
