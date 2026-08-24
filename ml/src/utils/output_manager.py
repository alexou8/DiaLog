"""Output management system for DiaLog."""
from pathlib import Path
from typing import Dict, Any, Optional
import json
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class OutputManager:
    """Manage model outputs, predictions, and metrics."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_predictions(
        self,
        predictions: pd.DataFrame,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Save predictions with metadata to CSV."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_predictions_{timestamp}.csv"
        filepath = self.output_dir / filename
        
        predictions.to_csv(filepath, index=False)
        
        # Save metadata as JSON
        if metadata:
            metadata_path = filepath.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Saved predictions to {filepath}")
        return filepath
    
    def save_metrics(
        self,
        metrics: Dict[str, float],
        model_name: str,
        run_id: Optional[str] = None
    ) -> Path:
        """Save evaluation metrics to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = run_id or timestamp
        filename = f"{model_name}_metrics_{run_id}.json"
        filepath = self.output_dir / filename
        
        metrics_with_meta = {
            'timestamp': timestamp,
            'model_name': model_name,
            'metrics': metrics
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics_with_meta, f, indent=2)
        
        return filepath
    
    def load_predictions(self, filepath: Path) -> pd.DataFrame:
        """Load predictions from CSV."""
        return pd.read_csv(filepath)
    
    def get_latest_predictions(self, model_name: str) -> Optional[pd.DataFrame]:
        """Get the most recent predictions for a given model."""
        pattern = f"{model_name}_predictions_*.csv"
        files = sorted(self.output_dir.glob(pattern), reverse=True)
        
        if files:
            return self.load_predictions(files[0])
        return None
