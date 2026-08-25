"""Utility functions for model management."""
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def get_latest_model(model_dir: Path, model_prefix: str) -> Optional[Path]:
    """
    Get the most recently saved model file matching the given prefix.
    
    Args:
        model_dir: Directory containing model files
        model_prefix: Prefix to match (e.g., 'glucose_predictor')
    
    Returns:
        Path to the latest model file, or None if not found
    """
    pattern = f"{model_prefix}*.pkl"
    model_files = sorted(model_dir.glob(pattern), reverse=True)
    
    if model_files:
        logger.info(f"Found {len(model_files)} model(s) matching '{pattern}'")
        return model_files[0]
    
    logger.warning(f"No models found matching '{pattern}' in {model_dir}")
    return None


def validate_model_config(config: dict) -> bool:
    """
    Validate that model configuration contains required fields.
    
    Args:
        config: Model configuration dictionary
    
    Returns:
        True if valid, False otherwise
    """
    required_fields = ['random_state']
    for field in required_fields:
        if field not in config:
            logger.error(f"Missing required config field: {field}")
            return False
    return True
