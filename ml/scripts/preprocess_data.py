"""Build the processed training-example dataset from raw ingested events.

Usage:
    python scripts/preprocess_data.py [--spike-threshold 200.0] [--lookback-hours 6.0]
"""
import argparse

from src.config import PATHS
from src.db import fetch_all_events
from src.features import FeatureConfig, build_training_examples, to_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spike-threshold", type=float, default=200.0, dest="spike_threshold_mgdl")
    parser.add_argument("--lookback-hours", type=float, default=6.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = FeatureConfig(spike_threshold_mgdl=args.spike_threshold_mgdl, lookback_hours=args.lookback_hours)

    rows = fetch_all_events()
    df = to_dataframe(rows)
    examples = build_training_examples(df, config=config)

    PATHS.DATA.mkdir(parents=True, exist_ok=True)
    examples.to_csv(PATHS.PROCESSED, index=False)
    print(f"Wrote processed dataset: {PATHS.PROCESSED} ({len(examples)} rows)")
