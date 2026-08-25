"""Unit tests for output management."""
import pytest
import pandas as pd
import json
from pathlib import Path
from src.utils.output_manager import OutputManager


@pytest.mark.unit
class TestOutputManager:
    """Test cases for OutputManager."""
    
    def test_initialization(self, temp_output_dir):
        """Test that OutputManager initializes correctly."""
        manager = OutputManager(temp_output_dir)
        
        assert manager.output_dir == temp_output_dir
        assert manager.output_dir.exists()
    
    def test_initialization_creates_directory(self, tmp_path):
        """Test that OutputManager creates output directory if it doesn't exist."""
        output_dir = tmp_path / "new_outputs"
        assert not output_dir.exists()
        
        manager = OutputManager(output_dir)
        assert output_dir.exists()
    
    def test_save_predictions(self, temp_output_dir):
        """Test saving predictions to CSV."""
        manager = OutputManager(temp_output_dir)
        
        predictions_df = pd.DataFrame({
            'timestamp': ['2026-01-01', '2026-01-02'],
            'predicted_glucose': [120.5, 135.2],
            'actual_glucose': [118.0, 140.0]
        })
        
        output_path = manager.save_predictions(
            predictions_df,
            model_name="test_model"
        )
        
        assert output_path.exists()
        assert output_path.suffix == '.csv'
        assert 'test_model_predictions_' in output_path.name
        
        # Verify contents
        loaded_df = pd.read_csv(output_path)
        pd.testing.assert_frame_equal(loaded_df, predictions_df)
    
    def test_save_predictions_with_metadata(self, temp_output_dir):
        """Test saving predictions with metadata."""
        manager = OutputManager(temp_output_dir)
        
        predictions_df = pd.DataFrame({
            'predicted': [120, 130, 140]
        })
        
        metadata = {
            'model_version': 'v1',
            'features': ['carbs', 'insulin'],
            'n_predictions': 3
        }
        
        output_path = manager.save_predictions(
            predictions_df,
            model_name="test_model",
            metadata=metadata
        )
        
        # Check metadata file exists
        metadata_path = output_path.with_suffix('.json')
        assert metadata_path.exists()
        
        # Verify metadata contents
        with open(metadata_path, 'r') as f:
            loaded_metadata = json.load(f)
        
        assert loaded_metadata == metadata
    
    def test_save_metrics(self, temp_output_dir):
        """Test saving metrics to JSON."""
        manager = OutputManager(temp_output_dir)
        
        metrics = {
            'rmse': 15.5,
            'mae': 12.3,
            'r2': 0.85
        }
        
        output_path = manager.save_metrics(
            metrics,
            model_name="test_model",
            run_id="test_run"
        )
        
        assert output_path.exists()
        assert output_path.suffix == '.json'
        assert 'test_model_metrics_test_run' in output_path.name
        
        # Verify contents
        with open(output_path, 'r') as f:
            loaded_data = json.load(f)
        
        assert loaded_data['model_name'] == "test_model"
        assert loaded_data['metrics'] == metrics
        assert 'timestamp' in loaded_data
    
    def test_save_metrics_auto_run_id(self, temp_output_dir):
        """Test that save_metrics auto-generates run_id if not provided."""
        manager = OutputManager(temp_output_dir)
        
        metrics = {'rmse': 10.0}
        output_path = manager.save_metrics(metrics, model_name="test_model")
        
        assert output_path.exists()
        # Filename should contain a timestamp-based run_id
        assert 'test_model_metrics_' in output_path.name
    
    def test_load_predictions(self, temp_output_dir):
        """Test loading predictions from CSV."""
        manager = OutputManager(temp_output_dir)
        
        # Create and save predictions
        original_df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        })
        
        output_path = manager.save_predictions(original_df, model_name="test")
        
        # Load and compare
        loaded_df = manager.load_predictions(output_path)
        pd.testing.assert_frame_equal(loaded_df, original_df)
    
    def test_get_latest_predictions(self, temp_output_dir):
        """Test getting the latest predictions for a model."""
        manager = OutputManager(temp_output_dir)
        
        # Save multiple predictions
        for i in range(3):
            df = pd.DataFrame({'value': [i]})
            manager.save_predictions(df, model_name="test_model")
        
        # Get latest
        latest_df = manager.get_latest_predictions("test_model")
        
        assert latest_df is not None
        assert len(latest_df) == 1
        # Latest should have value=2 (last saved)
        assert latest_df['value'].iloc[0] == 2
    
    def test_get_latest_predictions_no_files(self, temp_output_dir):
        """Test getting latest predictions when no files exist."""
        manager = OutputManager(temp_output_dir)
        
        latest_df = manager.get_latest_predictions("nonexistent_model")
        assert latest_df is None
