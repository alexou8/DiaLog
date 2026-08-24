# Testing Guide for DiaLog

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Sample Data
```bash
python examples/generate_sample_data.py
```

### 3. Train a Model
```bash
python examples/train_model_example.py
```

### 4. Make Predictions
```bash
python examples/make_predictions_example.py
```

### 5. Run Tests
```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit -v

# Run integration tests only
pytest tests/integration -v

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test markers
pytest -m unit
pytest -m integration
pytest -m ml
```

## Project Structure

```
DiaLog/
├── src/
│   ├── models/              # Model implementations
│   │   ├── base.py          # Base model interface
│   │   ├── predictor.py     # Glucose predictor
│   │   └── utils.py         # Model utilities
│   ├── utils/               # Utility functions
│   │   ├── config.py        # Configuration management
│   │   ├── output_manager.py # Output management
│   │   └── logging.py       # Logging setup
│   └── data/                # Data management
│       ├── loaders.py       # Data loading utilities
│       └── validators.py    # Data validation
├── tests/
│   ├── unit/                # Unit tests
│   │   ├── test_models.py
│   │   ├── test_outputs.py
│   │   └── test_model_properties.py
│   ├── integration/         # Integration tests
│   │   └── test_model_output_pipeline.py
│   ├── fixtures/            # Test fixtures
│   └── conftest.py          # Shared pytest fixtures
├── examples/                # Usage examples
│   ├── generate_sample_data.py
│   ├── train_model_example.py
│   └── make_predictions_example.py
├── models/                  # Saved model files
├── outputs/                 # Predictions and metrics
└── data/                    # Data files
```

## Testing Infrastructure

### Unit Tests
Unit tests validate individual components in isolation:
- **test_models.py**: Tests for model training, prediction, save/load
- **test_outputs.py**: Tests for output management and file operations
- **test_model_properties.py**: Property-based tests using Hypothesis

### Integration Tests
Integration tests validate end-to-end workflows:
- **test_model_output_pipeline.py**: Complete pipeline from data to predictions

### Test Fixtures
Shared fixtures are defined in `tests/conftest.py`:
- `random_seed`: Set reproducible random seed
- `sample_glucose_data`: Generate sample glucose monitoring data
- `sample_training_data`: Generate X, y training data
- `temp_model_dir`: Temporary directory for model files
- `temp_output_dir`: Temporary directory for outputs

## Sample Inputs and Outputs

### Sample Input (data/sample_glucose_data.csv)
Generated glucose monitoring data with features:
- `timestamp`: Date and time of measurement
- `glucose_mg_dl`: Glucose level in mg/dL
- `carbs_grams`: Carbohydrate intake in grams
- `insulin_units`: Insulin dosage in units
- `activity_minutes`: Physical activity in minutes
- `stress_level`: Stress level (1-5)
- `sleep_hours`: Hours of sleep

### Sample Outputs (outputs/)

#### Predictions (CSV)
- Filename format: `{model_name}_predictions_{timestamp}.csv`
- Contains: actual values, predicted values, errors

#### Metadata (JSON)
- Filename format: `{model_name}_predictions_{timestamp}.json`
- Contains: model version, features used, number of predictions

#### Metrics (JSON)
- Filename format: `{model_name}_metrics_{run_id}.json`
- Contains: RMSE, MAE, R², timestamp, model name

## Model Interface

All models inherit from `BaseModel` and implement:
- `train(X, y)`: Train the model
- `predict(X)`: Make predictions
- `save(path)`: Save model to disk
- `load(path)`: Load model from disk
- `evaluate(X, y)`: Evaluate and return metrics

## Configuration

The `Config` class provides centralized configuration:
```python
from src.utils.config import Config

config = Config()
print(config.MODELS_DIR)   # Path to models directory
print(config.OUTPUTS_DIR)  # Path to outputs directory
print(config.DATA_DIR)     # Path to data directory
```

Environment variables:
- `MODEL_VERSION`: Model version (default: "v1")
- `RANDOM_SEED`: Random seed (default: 42)
- `N_ESTIMATORS`: Number of trees (default: 100)
- `MAX_DEPTH`: Maximum tree depth (default: 10)

## Output Management

The `OutputManager` class handles all output operations:
```python
from src.utils.output_manager import OutputManager

manager = OutputManager(output_dir)

# Save predictions
manager.save_predictions(df, model_name="my_model", metadata={...})

# Save metrics
manager.save_metrics(metrics, model_name="my_model", run_id="run_1")

# Load predictions
df = manager.load_predictions(path)

# Get latest predictions
latest = manager.get_latest_predictions(model_name="my_model")
```

## CI/CD Pipeline

The project includes a GitHub Actions workflow (`.github/workflows/ci.yml`) that:
- Runs tests on Python 3.9, 3.10, and 3.11
- Executes unit and integration tests
- Generates coverage reports
- Uploads coverage to Codecov

## Development Workflow

1. **Make changes** to source code
2. **Run unit tests** to validate individual components:
   ```bash
   pytest tests/unit -v
   ```
3. **Run integration tests** to validate complete workflows:
   ```bash
   pytest tests/integration -v
   ```
4. **Check coverage** to ensure adequate test coverage:
   ```bash
   pytest --cov=src --cov-report=html
   open htmlcov/index.html
   ```
5. **Generate sample data** and run examples to validate functionality:
   ```bash
   python examples/generate_sample_data.py
   python examples/train_model_example.py
   python examples/make_predictions_example.py
   ```

## Best Practices

1. **Always use fixtures** for test data to ensure reproducibility
2. **Test edge cases** including empty inputs, invalid data
3. **Use temporary directories** in tests to avoid polluting actual folders
4. **Mock external dependencies** when necessary
5. **Keep tests fast** - use small datasets for unit tests
6. **Document complex tests** with clear docstrings
7. **Use parametrized tests** for testing multiple scenarios

## Troubleshooting

### Tests fail with import errors
Make sure you're running tests from the project root:
```bash
cd /path/to/DiaLog
pytest
```

### Missing sample data
Generate sample data first:
```bash
python examples/generate_sample_data.py
```

### Model not found errors in examples
Train the model before making predictions:
```bash
python examples/train_model_example.py
python examples/make_predictions_example.py
```

### Coverage reports not generated
Install pytest-cov:
```bash
pip install pytest-cov
```

## Contributing

When adding new features:
1. Add corresponding unit tests
2. Update integration tests if workflow changes
3. Update documentation
4. Ensure all tests pass: `pytest`
5. Check coverage: `pytest --cov=src`

## Additional Resources

- [pytest Documentation](https://docs.pytest.org/)
- [Hypothesis Documentation](https://hypothesis.readthedocs.io/)
- [scikit-learn Documentation](https://scikit-learn.org/)
