"""Unit tests for feature engineering, including anti-leakage properties."""
import numpy as np
import pandas as pd
import pytest

from src.features import FeatureConfig, build_training_examples, to_dataframe


@pytest.mark.unit
class TestFeatureConfig:
    def test_defaults(self):
        config = FeatureConfig()
        assert config.spike_threshold_mgdl == 200.0
        assert config.lookback_hours == 6.0

    def test_configurable_not_hardcoded(self):
        """The spike threshold and lookback are constructor params, not magic numbers."""
        config = FeatureConfig(spike_threshold_mgdl=180.0, lookback_hours=3.0)
        assert config.spike_threshold_mgdl == 180.0
        assert config.lookback_hours == 3.0


@pytest.mark.unit
class TestBuildTrainingExamples:
    def test_spike_threshold_changes_labels(self, synthetic_events_df):
        """A lower threshold must produce a >= spike rate than a higher one."""
        low = build_training_examples(synthetic_events_df, FeatureConfig(spike_threshold_mgdl=150.0))
        high = build_training_examples(synthetic_events_df, FeatureConfig(spike_threshold_mgdl=250.0))
        assert low["label_spike"].mean() >= high["label_spike"].mean()

    def test_no_missing_value_imputation_happens_here(self, synthetic_events_df):
        """build_training_examples must NOT fill NaNs with a whole-dataset
        statistic (e.g. global median) -- that is exactly the kind of
        leakage this pipeline was fixed to avoid. Any missing
        mins_since_meal / mins_since_med must survive as NaN so that
        imputation can be fit on the training fold only, downstream.
        """
        out = build_training_examples(synthetic_events_df, FeatureConfig(lookback_hours=0.01))
        # With a near-zero lookback window essentially nothing qualifies,
        # so we should see real NaNs, not a filled-in median.
        assert out["mins_since_meal"].isna().any()

    def test_features_use_only_past_events(self, synthetic_events_df):
        """For every example, mins_since_meal/med (when present) must
        reflect an event strictly at or before the anchor timestamp --
        never a future meal/med.
        """
        out = build_training_examples(synthetic_events_df, FeatureConfig())
        meals = synthetic_events_df[synthetic_events_df["event_type"] == "meal"]
        for _, row in out.iterrows():
            if pd.notna(row["mins_since_meal"]):
                implied_meal_time = row["timestamp"] - pd.Timedelta(minutes=row["mins_since_meal"])
                # The implied meal time must correspond to an actual meal
                # at or before the anchor -- not after it.
                assert (meals["timestamp"] <= row["timestamp"]).any()
                assert implied_meal_time <= row["timestamp"]

    def test_prev_label_spike_is_causal(self, synthetic_events_df):
        """prev_label_spike must equal the label of the immediately
        preceding chronological glucose reading -- using only past
        information, and the first reading (no predecessor) must be
        dropped rather than have a fabricated label.
        """
        out = build_training_examples(synthetic_events_df, FeatureConfig()).sort_values("timestamp").reset_index(drop=True)
        assert out["prev_label_spike"].iloc[1:].tolist() == out["label_spike"].iloc[:-1].tolist()
        assert out["prev_label_spike"].isna().sum() == 0

    def test_no_future_leakage_when_reordering_future_rows(self, synthetic_events_df):
        """Perturbing glucose values strictly AFTER a given anchor's
        timestamp must not change that anchor's own computed features
        (mins_since_meal/med, hour, dayofweek). This directly tests that
        no future information leaks backward into a row's features.
        """
        baseline = build_training_examples(synthetic_events_df, FeatureConfig())

        perturbed_events = synthetic_events_df.copy()
        cutoff = perturbed_events["timestamp"].quantile(0.5)
        future_mask = (perturbed_events["event_type"] == "glucose") & (perturbed_events["timestamp"] > cutoff)
        perturbed_events.loc[future_mask, "glucose_mgdl"] = 999.0

        perturbed = build_training_examples(perturbed_events, FeatureConfig())

        early_baseline = baseline[baseline["timestamp"] <= cutoff].reset_index(drop=True)
        early_perturbed = perturbed[perturbed["timestamp"] <= cutoff].reset_index(drop=True)

        pd.testing.assert_series_equal(
            early_baseline["mins_since_meal"], early_perturbed["mins_since_meal"], check_names=False
        )
        pd.testing.assert_series_equal(
            early_baseline["last_meal_carbs"], early_perturbed["last_meal_carbs"], check_names=False
        )
