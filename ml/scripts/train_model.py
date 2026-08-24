"""Train the spike classifier on the processed dataset and print an honest report.

Usage:
    python scripts/train_model.py [--spike-threshold 200.0] [--lookback-hours 6.0]
        [--data-provenance "..."]
"""
import argparse

import joblib
import pandas as pd

from src.config import PATHS
from src.modeling import train_and_evaluate
from src.features import FeatureConfig

DEFAULT_PROVENANCE = (
    "UNKNOWN -- run scripts/preprocess_data.py against your own event log, or "
    "pass --data-provenance explicitly. If this data traces back to "
    "ml/data/sample_glucose_data.csv / examples/generate_sample_data.py it is "
    "SYNTHETIC and carries no clinical validity."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spike-threshold", type=float, default=200.0, dest="spike_threshold_mgdl")
    parser.add_argument("--lookback-hours", type=float, default=6.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--data-provenance", type=str, default=DEFAULT_PROVENANCE)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    df = pd.read_csv(PATHS.PROCESSED)

    feature_config = FeatureConfig(
        spike_threshold_mgdl=args.spike_threshold_mgdl, lookback_hours=args.lookback_hours
    )
    report = train_and_evaluate(
        df,
        data_provenance=args.data_provenance,
        feature_config=feature_config,
        random_state=args.random_state,
    )

    PATHS.MODELS.mkdir(parents=True, exist_ok=True)
    model_path = PATHS.MODELS / "spike_classifier.joblib"
    joblib.dump(report.model, model_path)

    print("Model trained.")
    print(f"Data provenance: {report.data_provenance}")
    print(f"Sample sizes: {report.sample_sizes}")
    print(f"Class balance (positive rate): {report.class_balance}")
    print(f"ROC-AUC (test): {report.model_metrics.get('roc_auc', float('nan')):.3f}", end="")
    if "roc_auc" in report.model_metrics_ci:
        lo, hi = report.model_metrics_ci["roc_auc"]
        print(f"  95% CI [{lo:.3f}, {hi:.3f}]")
    else:
        print()
    print("Baseline comparisons (ROC-AUC):")
    for name, m in report.baseline_metrics.items():
        lift = report.lift_over_baselines[name].get("roc_auc", float("nan"))
        print(f"  - {name}: {m.get('roc_auc', float('nan')):.3f} (model lift: {lift:+.3f})")
    if report.warnings:
        print("Warnings:")
        for w in report.warnings:
            print(f"  - {w}")
    print()
    print(report.metrics_text)
    print(f"Saved model to: {model_path}")
