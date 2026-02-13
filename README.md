# DiaLog — Diabetes Tracking + ML Pattern Insights (Informational Only)

DiaLog is an AI-powered diabetes tracking and machine learning insight platform that analyzes
nutrition intake, medication timing, and glucose readings to surface **personalized, non-medical
pattern insights**.

An end-to-end ML pipeline that:
- stores diabetes-related logs (meals, meds, glucose) locally in SQLite
- preprocesses & engineers time-based features
- trains an ML classifier to estimate the probability of a "spike event"
- tunes hyperparameters
- generates weekly insights reports
- provides a modular, testable model abstraction layer
- includes comprehensive testing infrastructure with 32+ passing tests

## ⚠️ Disclaimer
- This project is for **informational and educational purposes only**.  
- It does **not** provide medical advice, diagnosis, treatment, or medication recommendations.
- All outputs are intended solely to highlight data patterns and trends.

## Key Features
- Structured logging of meals, medication events, and glucose readings
- Time-aware feature engineering (meal proximity, carb intake, medication timing)
- Machine learning–based glucose spike pattern detection
- Automated weekly insight and trend reporting
- Modular, maintainable codebase following software engineering best practices
- **Model Abstraction Layer**: Clean interface for model training, prediction, and persistence
- **Output Management System**: Centralized handling of predictions, metrics, and metadata
- **Comprehensive Testing**: 32+ unit and integration tests with property-based testing
- **CI/CD Pipeline**: Automated testing across Python 3.9, 3.10, and 3.11
- **Example Scripts**: Ready-to-use examples for data generation, training, and prediction

## Project Structure
```
DiaLog/
├── .github/
│   └── workflows/
│       └── ci.yml              # CI/CD pipeline configuration
├── src/                        # Core application logic
│   ├── models/                 # Model abstraction layer
│   │   ├── base.py            # Abstract base class for all models
│   │   ├── predictor.py       # Glucose prediction model
│   │   └── utils.py           # Model utilities
│   ├── utils/                  # Utility modules
│   │   ├── config.py          # Configuration management
│   │   ├── output_manager.py  # Output handling and persistence
│   │   └── logging.py         # Logging setup
│   ├── data/                   # Data management
│   │   ├── loaders.py         # Data loading utilities
│   │   └── validators.py      # Data validation
│   ├── db.py                  # Database operations
│   ├── features.py            # Feature engineering
│   ├── modeling.py            # ML modeling logic
│   └── reports.py             # Report generation
├── tests/                      # Test suite
│   ├── unit/                  # Unit tests (27 tests)
│   ├── integration/           # Integration tests (5 tests)
│   ├── fixtures/              # Test fixtures
│   └── conftest.py            # Shared test configuration
├── examples/                   # Usage examples
│   ├── generate_sample_data.py    # Generate sample glucose data
│   ├── train_model_example.py     # Model training example
│   └── make_predictions_example.py # Prediction example
├── data/                       # Sample and processed datasets
├── scripts/                    # Executable pipeline scripts
├── models/                     # Trained ML models
└── outputs/                    # Generated reports and predictions
```

## Machine Learning Overview
- **Problem Type:** Binary classification (glucose spike event detection)
- **Target Definition:** Glucose readings exceeding a configurable threshold
- **Feature Types:**
  - Temporal features (time of day, day of week)
  - Time since last meal / medication
  - Carbohydrate intake
  - Medication context
- **Model:** Random Forest classifier with class balancing
- **Evaluation:** ROC-AUC, precision, recall

The project prioritizes **interpretability, robustness, and data quality** over raw prediction
accuracy.

## Impact & Results
- **Glucose Spike Detection Performance:** Achieved a ROC-AUC score of **0.72–0.78** in identifying glucose spike events using time-series feature engineering and supervised classification.
- **Pattern Discovery:** Identified recurring high-risk periods (e.g., post-meal windows and time-of-day effects) that consistently correlated with elevated glucose readings.
- **Feature Insights:** Revealed that time since last meal, carbohydrate intake, and medication proximity were among the most influential features driving spike predictions.
- **Automation Efficiency:** Reduced manual trend analysis by generating automated weekly insight reports summarizing high-variability patterns and recurring risk windows.
- **End-to-End ML Pipeline:** Implemented a fully reproducible pipeline covering ingestion, preprocessing, training, evaluation, and reporting with no manual intervention.


## Tech Stack
- **Language:** Python  
- **Data & Feature Engineering:**  
  - `pandas` – structured data manipulation and time-series processing  
  - `NumPy` – numerical computation and feature transformation  

- **Machine Learning:**  
  - `scikit-learn` – supervised classification, model evaluation, and hyperparameter tuning  
  - `RandomForestClassifier` – non-linear pattern learning and feature importance analysis  

- **Data Storage:**  
  - `SQLite` – lightweight relational database for event logging  

- **Model Management:**  
  - `joblib` – model serialization and reuse  

- **Testing & Quality Assurance:**  
  - `pytest` – comprehensive testing framework  
  - `pytest-cov` – code coverage reporting  
  - `hypothesis` – property-based testing  

- **CI/CD:**  
  - GitHub Actions – automated testing pipeline  
  - Multi-version Python testing (3.9, 3.10, 3.11)  

- **Automation & Reporting:**  
  - Script-driven execution for preprocessing, training, tuning, and weekly insight generation  

- **Development Practices:**  
  - Modular architecture with separation of ingestion, modeling, and reporting layers  
  - Abstract base classes for model extensibility  
  - Centralized output management for predictions and metrics  
  - Reproducible, script-based ML workflows suitable for production extension

## Quickstart

### Option 1: Using Example Scripts (Recommended for New Users)
```bash
# 1. Create a virtual env and install dependencies
pip install -r requirements.txt

# 2. Generate sample glucose monitoring data
python examples/generate_sample_data.py

# 3. Train a machine learning model
python examples/train_model_example.py

# 4. Make predictions with the trained model
python examples/make_predictions_example.py

# 5. Run the test suite
pytest
```

### Option 2: Using the Original Pipeline Scripts
```bash
# 1. Create a virtual env and install dependencies
pip install -r requirements.txt

# 2. Initialize the database
python scripts/init_db.py

# 3. Ingest sample data
python scripts/ingest_csv.py data/sample_logs.csv

# 4. Preprocess and generate ML dataset
python scripts/preprocess_data.py

# 5. Train the machine learning model
python scripts/train_model.py

# 6. (Optional) Tune hyperparameters
python scripts/tune_hyperparams.py

# 7. Generate weekly insight report
python scripts/weekly_report.py
```

## Testing & Quality Assurance

DiaLog includes a comprehensive testing infrastructure to ensure reliability and maintainability:

### Test Coverage
- **32+ Tests**: 27 unit tests + 5 integration tests
- **Test Coverage**: 28% overall (100% for core modules like predictor.py and output_manager.py)
- **Property-Based Testing**: Using Hypothesis for comprehensive edge-case validation
- **CI/CD**: Automated testing across Python 3.9, 3.10, and 3.11 via GitHub Actions

### Running Tests
```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit -v

# Run integration tests only
pytest tests/integration -v

# Run with coverage report
pytest --cov=src --cov-report=html
```

For more details, see [README_TESTING.md](README_TESTING.md).

## Model Abstraction Layer

DiaLog features a clean, extensible model abstraction layer:

### BaseModel Interface
All models inherit from `BaseModel` and implement:
- `train(X, y)`: Train the model on provided data
- `predict(X)`: Make predictions on new data
- `save(path)`: Persist model to disk
- `load(path)`: Load model from disk
- `evaluate(X, y)`: Evaluate model performance and return metrics

### GlucosePredictor
The primary implementation using Random Forest:
- Configurable hyperparameters (n_estimators, max_depth, random_state)
- Automatic metric calculation (RMSE, MAE, R²)
- Pickle-based persistence for easy deployment

### Output Management
The `OutputManager` class provides centralized output handling:
- Save predictions with metadata (timestamps, model version, features used)
- Save evaluation metrics with run IDs
- Load and retrieve predictions
- Query latest predictions by model name

Example usage:
```python
from src.models.predictor import GlucosePredictor
from src.utils.output_manager import OutputManager

# Initialize model and output manager
model = GlucosePredictor(n_estimators=100, max_depth=10)
output_mgr = OutputManager("outputs")

# Train and evaluate
model.train(X_train, y_train)
metrics = model.evaluate(X_test, y_test)

# Save outputs
model.save("models/glucose_model.pkl")
output_mgr.save_metrics(metrics, model_name="glucose_model", run_id="run_1")
```

## Data format (CSV)
See data/sample_logs.csv
