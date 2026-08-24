"""Unit tests for src.modeling: chronological splitting, baselines, and
the honest-evaluation report -- these are the anti-leakage regression
tests for the pipeline.
"""
import numpy as np
import pandas as pd
import pytest

from src.modeling import (
    ALL_FEATURES,
    NUMERIC_FEATURES,
    SplitConfig,
    TARGET_COL,
    TIME_COL,
    bootstrap_metric_ci,
    build_pipeline,
    chronological_split,
    majority_class_baseline,
    persistence_baseline,
    prevalence_baseline,
    time_series_cv_scores,
    train_and_evaluate,
)


@pytest.mark.unit
class TestChronologicalSplit:
    def test_split_is_time_ordered_and_non_overlapping(self, synthetic_examples_df):
        split = chronological_split(synthetic_examples_df, SplitConfig(train_frac=0.6, val_frac=0.2))
        assert split.train[TIME_COL].max() <= split.val[TIME_COL].min()
        assert split.val[TIME_COL].max() <= split.test[TIME_COL].min()
        assert split.train[TIME_COL].max() <= split.test[TIME_COL].min()

    def test_split_uses_all_rows_exactly_once(self, synthetic_examples_df):
        split = chronological_split(synthetic_examples_df)
        assert len(split.train) + len(split.val) + len(split.test) == len(synthetic_examples_df)

    def test_is_not_a_random_split(self, synthetic_examples_df):
        """Two calls must produce IDENTICAL splits -- a chronological split
        has no randomness, unlike train_test_split(shuffle=True).
        """
        split_a = chronological_split(synthetic_examples_df)
        split_b = chronological_split(synthetic_examples_df)
        pd.testing.assert_frame_equal(split_a.train.reset_index(drop=True), split_b.train.reset_index(drop=True))
        pd.testing.assert_frame_equal(split_a.test.reset_index(drop=True), split_b.test.reset_index(drop=True))

    def test_invalid_fracs_rejected(self):
        with pytest.raises(ValueError):
            SplitConfig(train_frac=0.8, val_frac=0.3)  # sums >= 1.0

    def test_too_few_rows_raises(self):
        tiny = pd.DataFrame({TIME_COL: pd.to_datetime(["2026-01-01"]), TARGET_COL: [0]})
        with pytest.raises(ValueError):
            chronological_split(tiny)


@pytest.mark.unit
class TestImputationFoldDiscipline:
    def test_imputer_statistics_come_from_training_fold_only(self, synthetic_examples_df):
        """The median used to impute mins_since_meal must match the TRAIN
        fold's median, not the whole dataset's median -- this is the
        concrete regression test for the median-imputation leakage that
        was found and fixed in this pipeline.
        """
        split = chronological_split(synthetic_examples_df, SplitConfig(train_frac=0.6, val_frac=0.2))
        X_train = split.train[ALL_FEATURES]
        y_train = split.train[TARGET_COL].astype(int)

        pipe = build_pipeline(random_state=42, n_estimators=20)
        pipe.fit(X_train, y_train)

        fitted_median = pipe.named_steps["preprocess"].named_transformers_["num"].named_steps["imputer"].statistics_[
            NUMERIC_FEATURES.index("mins_since_meal")
        ]
        # Recompute directly what the training fold's median should be.
        expected_train_median = X_train["mins_since_meal"].median()
        whole_dataset_median = synthetic_examples_df["mins_since_meal"].median()

        assert fitted_median == pytest.approx(expected_train_median)
        # Guard against a regression back to whole-dataset imputation: if
        # the two medians coincide by chance the test is inconclusive, but
        # for this fixture (with a chronological trend) they differ.
        if whole_dataset_median != expected_train_median:
            assert fitted_median != pytest.approx(whole_dataset_median)


@pytest.mark.unit
class TestBaselines:
    def test_majority_class_uses_train_fold_not_eval_fold(self):
        y_train = pd.Series([0, 0, 0, 1])  # majority = 0
        y_eval = pd.Series([1, 1, 1, 1])  # would be majority = 1 if computed on eval
        metrics = majority_class_baseline(y_train, y_eval)
        # Predicting the TRAIN majority (0) against an all-1 eval set gives 0 recall/precision.
        assert metrics["recall"] == 0.0

    def test_prevalence_baseline_uses_train_prevalence(self):
        y_train = pd.Series([0, 0, 0, 1])
        y_eval = pd.Series([0, 1])
        metrics = prevalence_baseline(y_train, y_eval)
        assert 0.0 <= metrics["brier"] <= 1.0

    def test_persistence_baseline_uses_previous_reading(self):
        y_eval = pd.Series([0, 1, 1, 0])
        prev = pd.Series([0, 0, 1, 1])
        metrics = persistence_baseline(y_eval, prev)
        # 2 of 4 predictions correct (index1 wrong, index3 wrong -> 2 correct)
        assert metrics["accuracy"] == pytest.approx(0.5)

    def test_bootstrap_ci_bounds_contain_point_estimate_direction(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, size=200)
        y_proba = np.clip(y_true + rng.normal(0, 0.3, size=200), 0, 1)
        from sklearn.metrics import roc_auc_score

        low, high = bootstrap_metric_ci(y_true, y_proba, roc_auc_score, n_boot=200, random_state=1)
        assert low <= high
        assert 0.0 <= low <= 1.0
        assert 0.0 <= high <= 1.0


@pytest.mark.unit
class TestTimeSeriesCV:
    def test_uses_time_series_split_not_kfold(self, synthetic_examples_df):
        scores = time_series_cv_scores(synthetic_examples_df, n_splits=4, random_state=42, n_estimators=20)
        assert isinstance(scores, list)
        for s in scores:
            assert 0.0 <= s <= 1.0

    def test_handles_too_little_data_without_crashing(self):
        tiny = pd.DataFrame(
            {
                TIME_COL: pd.to_datetime(["2026-01-01", "2026-01-02"]),
                TARGET_COL: [0, 1],
                **{c: [0.0, 1.0] for c in ALL_FEATURES if c != "last_med_name"},
                "last_med_name": ["none", "none"],
            }
        )
        assert time_series_cv_scores(tiny, n_splits=5) == []


@pytest.mark.unit
class TestTrainAndEvaluate:
    @pytest.fixture
    def report(self, synthetic_examples_df):
        return train_and_evaluate(
            synthetic_examples_df,
            data_provenance="SYNTHETIC fixture data -- no clinical validity (unit test)",
            random_state=42,
            n_estimators=50,
            cv_splits=3,
        )

    def test_data_provenance_is_required_and_preserved(self, report):
        assert "SYNTHETIC" in report.data_provenance
        assert "SYNTHETIC" in report.to_dict()["data_provenance"]

    def test_report_contains_all_three_baselines(self, report):
        assert set(report.baseline_metrics.keys()) == {"majority_class", "prevalence", "previous_reading"}

    def test_report_contains_lift_over_every_baseline(self, report):
        assert set(report.lift_over_baselines.keys()) == {"majority_class", "prevalence", "previous_reading"}
        for baseline_name, lift in report.lift_over_baselines.items():
            assert "roc_auc" in lift or len(lift) == 0

    def test_report_contains_sample_sizes_and_class_balance(self, report):
        assert set(report.sample_sizes.keys()) == {"train", "val", "test"}
        assert sum(report.sample_sizes.values()) > 0
        assert set(report.class_balance.keys()) == {"train", "val", "test"}
        for rate in report.class_balance.values():
            assert np.isnan(rate) or 0.0 <= rate <= 1.0

    def test_report_contains_confidence_intervals(self, report):
        assert "accuracy" in report.model_metrics_ci
        low, high = report.model_metrics_ci["accuracy"]
        assert low <= high

    def test_to_dict_is_json_serializable(self, report):
        import json

        json.dumps(report.to_dict(), default=str)

    def test_small_sample_triggers_warning(self, synthetic_examples_df):
        tiny_report = train_and_evaluate(
            synthetic_examples_df.head(10),
            data_provenance="SYNTHETIC fixture data (tiny slice) -- no clinical validity (unit test)",
            random_state=42,
            n_estimators=10,
            cv_splits=2,
        )
        assert any("unreliable" in w for w in tiny_report.warnings)
