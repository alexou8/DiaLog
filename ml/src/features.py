"""Feature engineering for the glucose-spike research pipeline.

All features here are computed strictly from information available at or
before each glucose reading's timestamp (causal / no look-ahead). Missing
value imputation is intentionally NOT done in this module: fitting an
imputer (e.g. a median) on the full dataset before a train/val/test split
leaks information about validation/test rows into training. Imputation is
instead fit fold-by-fold inside the modeling pipeline (see ``src/modeling.py``),
using statistics from the training fold only.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Typical realistic glucose sensor range; used only for sanity checks upstream.
DEFAULT_SPIKE_THRESHOLD_MGDL: float = 200.0
DEFAULT_LOOKBACK_HOURS: float = 6.0


@dataclass(frozen=True)
class FeatureConfig:
    """Configurable parameters for feature construction.

    These were previously hardcoded magic numbers scattered through the
    pipeline; centralizing them here makes the spike definition and the
    meal/med attribution window explicit and overridable (e.g. from a
    script's CLI args or a hyperparameter search).
    """

    spike_threshold_mgdl: float = DEFAULT_SPIKE_THRESHOLD_MGDL
    lookback_hours: float = DEFAULT_LOOKBACK_HOURS


def to_dataframe(db_rows: list[tuple]) -> pd.DataFrame:
    df = pd.DataFrame(
        db_rows,
        columns=["timestamp", "event_type", "carbs_g", "med_name", "med_units", "glucose_mgdl", "notes"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def build_training_examples(
    df: pd.DataFrame,
    config: FeatureConfig | None = None,
    *,
    spike_threshold_mgdl: float | None = None,
    lookback_hours: float | None = None,
) -> pd.DataFrame:
    """
    Create examples anchored at each glucose reading.

    Features: time since last meal, carbs in last meal, time since last med,
    med units, hour-of-day, day-of-week, and the previous glucose reading's
    spike label (used later as the persistence baseline). Label: spike_event
    = glucose >= threshold.

    Every feature is computed using only events at or before the anchor
    timestamp -- no future information is used.

    ``config`` carries the spike threshold and lookback window. The
    ``spike_threshold_mgdl`` / ``lookback_hours`` keyword arguments are kept
    for backward compatibility and, if given, override ``config``.

    Note: NaNs for "no qualifying event in the lookback window" are left as
    NaN (not imputed here) -- see module docstring.
    """
    config = config or FeatureConfig()
    threshold = spike_threshold_mgdl if spike_threshold_mgdl is not None else config.spike_threshold_mgdl
    lookback_h = lookback_hours if lookback_hours is not None else config.lookback_hours

    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek

    meals = df[df["event_type"] == "meal"][["timestamp", "carbs_g"]].copy()
    meds = df[df["event_type"] == "med"][["timestamp", "med_name", "med_units"]].copy()
    glc = df[df["event_type"] == "glucose"][["timestamp", "glucose_mgdl", "hour", "dayofweek"]].copy()
    glc = glc.sort_values("timestamp").reset_index(drop=True)

    glc["label_spike"] = (glc["glucose_mgdl"] >= threshold).astype(int)
    # Persistence baseline feature: the previous chronological reading's
    # label. This uses only past information relative to the anchor, so it
    # is not leakage -- it is exactly the "naive forecast" a clinician-style
    # persistence baseline would use.
    glc["prev_label_spike"] = glc["label_spike"].shift(1)

    lookback = pd.Timedelta(hours=lookback_h)

    meals = meals.sort_values("timestamp")
    meds = meds.sort_values("timestamp")
    meal_times = meals["timestamp"].to_numpy()
    meal_carbs = meals["carbs_g"].fillna(0).to_numpy()
    med_times = meds["timestamp"].to_numpy()
    med_units = meds["med_units"].fillna(0).to_numpy()
    med_names = meds["med_name"].fillna("unknown").to_numpy()

    def last_event_before(ts: pd.Timestamp, times_array: np.ndarray) -> int | None:
        # np.searchsorted needs a numpy datetime64 scalar, not a pandas
        # Timestamp, to compare against a datetime64[ns] array reliably
        # across numpy/pandas versions.
        idx = np.searchsorted(times_array, np.datetime64(ts)) - 1
        return idx if idx >= 0 else None

    rows = []
    for _, r in glc.iterrows():
        ts = r["timestamp"]

        mi = last_event_before(ts, meal_times)
        if mi is not None and (ts - meal_times[mi]) <= lookback:
            mins_since_meal = (ts - meal_times[mi]).total_seconds() / 60.0
            last_meal_carbs = float(meal_carbs[mi]) if not pd.isna(meal_carbs[mi]) else 0.0
        else:
            mins_since_meal = np.nan
            last_meal_carbs = 0.0

        mdi = last_event_before(ts, med_times)
        if mdi is not None and (ts - med_times[mdi]) <= lookback:
            mins_since_med = (ts - med_times[mdi]).total_seconds() / 60.0
            last_med_units = float(med_units[mdi]) if not pd.isna(med_units[mdi]) else 0.0
            last_med_name = str(med_names[mdi]) if med_names[mdi] else "unknown"
        else:
            mins_since_med = np.nan
            last_med_units = 0.0
            last_med_name = "none"

        rows.append(
            {
                "timestamp": ts,
                "hour": int(r["hour"]),
                "dayofweek": int(r["dayofweek"]),
                "mins_since_meal": mins_since_meal,
                "last_meal_carbs": last_meal_carbs,
                "mins_since_med": mins_since_med,
                "last_med_units": last_med_units,
                "last_med_name": last_med_name,
                "glucose_mgdl": float(r["glucose_mgdl"]),
                "prev_label_spike": r["prev_label_spike"],
                "label_spike": int(r["label_spike"]),
            }
        )

    out = pd.DataFrame(rows)
    # Row 0 has no previous reading; drop it rather than impute a label --
    # imputing a class label would fabricate ground truth.
    out = out.dropna(subset=["prev_label_spike"]).reset_index(drop=True)
    out["prev_label_spike"] = out["prev_label_spike"].astype(int)
    return out
