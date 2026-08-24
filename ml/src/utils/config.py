"""Configuration management for DiaLog."""
from pathlib import Path
import os


class Config:
    """Centralized configuration for DiaLog."""
    
    def __init__(self):
        self.BASE_DIR = Path(__file__).parent.parent.parent
        self.MODELS_DIR = self.BASE_DIR / "models"
        self.OUTPUTS_DIR = self.BASE_DIR / "outputs"
        self.DATA_DIR = self.BASE_DIR / "data"
        
        # Model settings
        self.MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
        self.RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
        
        # Model hyperparameters
        self.N_ESTIMATORS = int(os.getenv("N_ESTIMATORS", "100"))
        self.MAX_DEPTH = int(os.getenv("MAX_DEPTH", "10"))
        
        # Create directories
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
