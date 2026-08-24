"""Base model interface for DiaLog models."""
from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np
from pathlib import Path


class BaseModel(ABC):
    """Abstract base class for all DiaLog models."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.is_trained = False
    
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model on input features X and target y."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on input features X."""
        pass
    
    @abstractmethod
    def save(self, path: Path) -> None:
        """Save the trained model to disk."""
        pass
    
    @abstractmethod
    def load(self, path: Path) -> None:
        """Load a trained model from disk."""
        pass
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance and return metrics."""
        predictions = self.predict(X)
        return self._compute_metrics(y, predictions)
    
    @abstractmethod
    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute evaluation metrics (RMSE, MAE, R2, etc.)."""
        pass
