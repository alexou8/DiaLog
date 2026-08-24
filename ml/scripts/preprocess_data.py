from src.db import fetch_all_events
from src.features import to_dataframe, build_training_examples
from src.config import PATHS

if __name__ == "__main__":
    rows = fetch_all_events()
    df = to_dataframe(rows)
    examples = build_training_examples(df, spike_threshold_mgdl=200.0, lookback_hours=6.0)

    PATHS.DATA.mkdir(parents=True, exist_ok=True)
    examples.to_csv(PATHS.PROCESSED, index=False)
    print(f"Wrote processed dataset: {PATHS.PROCESSED} ({len(examples)} rows)")
