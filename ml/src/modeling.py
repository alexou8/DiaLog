"""Training and honest evaluation for the glucose-spike research model.

This module is deliberately conservative about what it reports: every
headline metric is accompanied by a baseline comparison, a sample size, a
class-balance figure, and a bootstrap confidence interval, and the split
used for evaluation is chronological (train on the earliest data, evaluate
on later data) rather than a random shuffle, because the underlying data is
a time series and a random split leaks future information into training.

Nothing in this module is clinically validated. See docs/ML_PIPELINE.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .features import FeatureConfig

NUMERIC_FEATURES: list[str] = [
    "hour",
    "dayofweek",
    "mins_since_meal",
    "last_meal_carbs",
    "mins_since_med",
    "last_med_units",
]
CATEGORICAL_FEATURES: list[str] = ["last_med_name"]
ALL_FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COL: str = "label_spike"
TIME_COL: str = "timestamp"
PREV_LABEL_COL: str = "prev_label_spike"


@dataclass(frozen=True)
class SplitConfig:
    """Fractions for a chronological train/val/test split. Must sum to <= 1."""

    train_frac: float = 0.6
    val_frac: float = 0.2
    # test_frac is whatever remains (default 0.2)

    def __post_init__(self) -> None:
        if not (0.0 < self.train_frac < 1.0):
            raise ValueError("train_frac must be in (0, 1)")
        if not (0.0 <= self.val_frac < 1.0):
            raise ValueError("val_frac must be in [0, 1)")
        if self.train_frac + self.val_frac >= 1.0:
            raise ValueError("train_frac + val_frac must leave a nonzero test fraction")


@dataclass(frozen=True)
class ChronologicalSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def chronological_split(
    df: pd.DataFrame,
    split_config: SplitConfig | None = None,
    time_col: str = TIME_COL,
) -> ChronologicalSplit:
    """Split a time-ordered dataframe into train/val/test blocks by time.

    The dataframe is sorted by ``time_col`` first, then cut into three
    contiguous blocks in chronological order: training data is strictly the
    earliest rows, validation the middle rows, and test the most recent
    rows. This is the correct way to evaluate a model meant to predict
    forward in time -- unlike a random/stratified split, it cannot leak
    information from the future into training or hyperparameter selection.
    """
    split_config = split_config or SplitConfig()
    ordered = df.sort_values(time_col).reset_index(drop=True)
    n = len(ordered)
    if n < 3:
        raise ValueError(f"Need at least 3 rows for a train/val/test split, got {n}")

    train_end = int(n * split_config.train_frac)
    val_end = train_end + int(n * split_config.val_frac)
    train_end = max(1, min(train_end, n - 2))
    val_end = max(train_end + 1, min(val_end, n - 1))

    train_df = ordered.iloc[:train_end]
    val_df = ordered.iloc[train_end:val_end]
    test_df = ordered.iloc[val_end:]

    # Anti-leakage guard: every train timestamp must precede every test
    # timestamp (and val sit strictly between). Assert rather than silently
    # trust the arithmetic above.
    if len(train_df) and len(test_df):
        assert train_df[time_col].max() <= test_df[time_col].min(), (
            "chronological_split produced overlapping train/test time ranges"
        )
    if len(val_df) and len(test_df):
        assert val_df[time_col].max() <= test_df[time_col].min(), (
            "chronological_split produced overlapping val/test time ranges"
        )

    return ChronologicalSplit(train=train_df, val=val_df, test=test_df)


def build_pipeline(random_state: int = 42, n_estimators: int = 300) -> Pipeline:
    """Build the preprocessing + model pipeline.

    Imputation (median for numeric, most-frequent for categorical) is fit
    inside this pipeline's ``.fit()`` call, i.e. on whatever data is passed
    in -- the training fold only, when used correctly by callers in this
    module. This is what makes it leakage-free, in contrast to imputing on
    the whole dataset before splitting.
    """
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), NUMERIC_FEATURES),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=random_state,
        class_weight="balanced",
    )
    return Pipeline(steps=[("preprocess", pre), ("clf", clf)])


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        metrics["brier"] = float(brier_score_loss(y_true, y_proba))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["brier"] = float(np.mean((y_proba - y_true) ** 2)) if y_proba is not None else float("nan")
    return metrics


def majority_class_baseline(y_train: pd.Series, y_eval: pd.Series) -> dict[str, float]:
    """Predict the majority class observed in the TRAINING fold for every eval row."""
    majority = int(y_train.mode().iloc[0]) if len(y_train) else 0
    y_pred = np.full(len(y_eval), majority)
    y_proba = np.full(len(y_eval), float(majority))
    return _classification_metrics(y_eval.to_numpy(), y_pred, y_proba)


def prevalence_baseline(y_train: pd.Series, y_eval: pd.Series) -> dict[str, float]:
    """Predict the TRAINING fold's positive-class prevalence as a constant probability for every eval row."""
    prevalence = float(y_train.mean()) if len(y_train) else 0.0
    y_proba = np.full(len(y_eval), prevalence)
    y_pred = (y_proba >= 0.5).astype(int)
    return _classification_metrics(y_eval.to_numpy(), y_pred, y_proba)


def persistence_baseline(y_eval: pd.Series, prev_label_eval: pd.Series) -> dict[str, float]:
    """Predict that the spike label repeats from the immediately preceding reading.

    This is the "naive forecast" every real predictive-monitoring baseline
    must beat: it uses only the previous reading, no model, no training.
    """
    y_pred = prev_label_eval.to_numpy().astype(int)
    y_proba = y_pred.astype(float)
    return _classification_metrics(y_eval.to_numpy(), y_pred, y_proba)


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------

def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric_fn,
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> tuple[float, float]:
    """Bootstrap a (1 - alpha) confidence interval for a metric over the eval set.

    Resamples (with replacement) rows of the evaluation set n_boot times and
    recomputes the metric each time. Returns (low, high) percentile bounds.
    Skips resamples that end up single-class (undefined for e.g. ROC-AUC).
    """
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    if n == 0:
        return (float("nan"), float("nan"))
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_proba[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            scores.append(metric_fn(yt, yp))
        except ValueError:
            continue
    if not scores:
        return (float("nan"), float("nan"))
    low = float(np.percentile(scores, 100 * (alpha / 2)))
    high = float(np.percentile(scores, 100 * (1 - alpha / 2)))
    return (low, high)


# ---------------------------------------------------------------------------
# Cross-validation for hyperparameter tuning / robustness reporting
# ---------------------------------------------------------------------------

def time_series_cv_scores(
    df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
    n_estimators: int = 300,
) -> list[float]:
    """Chronological cross-validated ROC-AUC using sklearn's TimeSeriesSplit.

    TimeSeriesSplit always trains on an earlier contiguous block and
    evaluates on a later one, for every fold -- unlike KFold, it never
    trains on data that is chronologically after the evaluation fold.
    """
    ordered = df.sort_values(TIME_COL).reset_index(drop=True)
    X = ordered[ALL_FEATURES]
    y = ordered[TARGET_COL].astype(int)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores: list[float] = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            continue
        pipe = build_pipeline(random_state=random_state, n_estimators=n_estimators)
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        scores.append(float(roc_auc_score(y_test, proba)))
    return scores


# ---------------------------------------------------------------------------
# End-to-end honest evaluation
# ---------------------------------------------------------------------------

@dataclass
class EvaluationReport:
    model: Pipeline
    feature_config: FeatureConfig
    split_config: SplitConfig
    data_provenance: str
    sample_sizes: dict[str, int]
    class_balance: dict[str, float]
    model_metrics: dict[str, float]
    model_metrics_ci: dict[str, tuple[float, float]]
    baseline_metrics: dict[str, dict[str, float]]
    lift_over_baselines: dict[str, dict[str, float]]
    time_series_cv_roc_auc: list[float]
    metrics_text: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "data_provenance": self.data_provenance,
            "feature_config": {
                "spike_threshold_mgdl": self.feature_config.spike_threshold_mgdl,
                "lookback_hours": self.feature_config.lookback_hours,
            },
            "split_config": {
                "train_frac": self.split_config.train_frac,
                "val_frac": self.split_config.val_frac,
                "method": "chronological",
            },
            "sample_sizes": self.sample_sizes,
            "class_balance_positive_rate": self.class_balance,
            "model_metrics_on_test": self.model_metrics,
            "model_metrics_95ci_on_test": {k: list(v) for k, v in self.model_metrics_ci.items()},
            "baseline_metrics_on_test": self.baseline_metrics,
            "lift_over_baselines": self.lift_over_baselines,
            "time_series_cv_roc_auc": self.time_series_cv_roc_auc,
            "warnings": self.warnings,
        }


# Minimum test-set size below which metrics are flagged as unreliable.
MIN_RELIABLE_TEST_SIZE: int = 30


def train_and_evaluate(
    df: pd.DataFrame,
    data_provenance: str,
    feature_config: FeatureConfig | None = None,
    split_config: SplitConfig | None = None,
    random_state: int = 42,
    n_estimators: int = 300,
    cv_splits: int = 5,
) -> EvaluationReport:
    """Train the spike classifier and evaluate it honestly against baselines.

    ``data_provenance`` is required (not defaulted) so that every caller is
    forced to state, explicitly, where the underlying data came from (e.g.
    "SYNTHETIC sample data generated by examples/generate_sample_data.py --
    no clinical validity" vs. a real, clinically-validated dataset). This
    string is threaded straight into the report / export JSON.
    """
    feature_config = feature_config or FeatureConfig()
    split_config = split_config or SplitConfig()

    split = chronological_split(df, split_config)
    warnings: list[str] = []

    X_train, y_train = split.train[ALL_FEATURES], split.train[TARGET_COL].astype(int)
    X_test, y_test = split.test[ALL_FEATURES], split.test[TARGET_COL].astype(int)
    prev_label_test = split.test[PREV_LABEL_COL]

    if len(X_test) < MIN_RELIABLE_TEST_SIZE:
        warnings.append(
            f"Test set has only {len(X_test)} rows (< {MIN_RELIABLE_TEST_SIZE}); "
            "metrics and confidence intervals are unreliable at this sample size."
        )
    if y_train.nunique() < 2:
        warnings.append("Training fold contains a single class; model cannot learn to discriminate.")

    pipe = build_pipeline(random_state=random_state, n_estimators=n_estimators)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else y_pred.astype(float)

    model_metrics = _classification_metrics(y_test.to_numpy(), y_pred, y_proba)

    model_metrics_ci: dict[str, tuple[float, float]] = {}
    if y_test.nunique() > 1:
        model_metrics_ci["roc_auc"] = bootstrap_metric_ci(
            y_test.to_numpy(), y_proba, roc_auc_score, random_state=random_state
        )
    model_metrics_ci["accuracy"] = bootstrap_metric_ci(
        y_test.to_numpy(), y_pred.astype(float), lambda a, b: accuracy_score(a, b), random_state=random_state
    )

    baseline_metrics = {
        "majority_class": majority_class_baseline(y_train, y_test),
        "prevalence": prevalence_baseline(y_train, y_test),
        "previous_reading": persistence_baseline(y_test, prev_label_test),
    }

    lift_over_baselines: dict[str, dict[str, float]] = {}
    for name, bmetrics in baseline_metrics.items():
        lift_over_baselines[name] = {
            metric: (model_metrics[metric] - bmetrics[metric])
            for metric in model_metrics
            if not (np.isnan(model_metrics[metric]) or np.isnan(bmetrics[metric]))
        }

    cv_scores = time_series_cv_scores(
        pd.concat([split.train, split.val], axis=0),
        n_splits=cv_splits,
        random_state=random_state,
        n_estimators=n_estimators,
    )

    class_balance = {
        "train": float(y_train.mean()) if len(y_train) else float("nan"),
        "val": float(split.val[TARGET_COL].astype(int).mean()) if len(split.val) else float("nan"),
        "test": float(y_test.mean()) if len(y_test) else float("nan"),
    }
    sample_sizes = {"train": len(split.train), "val": len(split.val), "test": len(split.test)}

    report_text = classification_report(y_test, y_pred, zero_division=0)

    return EvaluationReport(
        model=pipe,
        feature_config=feature_config,
        split_config=split_config,
        data_provenance=data_provenance,
        sample_sizes=sample_sizes,
        class_balance=class_balance,
        model_metrics=model_metrics,
        model_metrics_ci=model_metrics_ci,
        baseline_metrics=baseline_metrics,
        lift_over_baselines=lift_over_baselines,
        time_series_cv_roc_auc=cv_scores,
        metrics_text=report_text,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Backward-compatible thin wrapper (kept so existing scripts / callers that
# only want "a trained model" still work without pulling in the full report)
# ---------------------------------------------------------------------------

@dataclass
class TrainResult:
    model: Pipeline
    metrics_text: str
    roc_auc: float
    report: EvaluationReport


def train_classifier(
    df: pd.DataFrame,
    data_provenance: str = "UNLABELED data source -- provenance not specified by caller",
    random_state: int = 42,
) -> TrainResult:
    """Train + honestly evaluate the spike classifier; return the trained
    pipeline plus its full evaluation report (baselines, CIs, provenance).

    Callers that only used to read ``.model``, ``.metrics_text`` and
    ``.roc_auc`` keep working; ``.report`` carries everything new.
    """
    report = train_and_evaluate(df, data_provenance=data_provenance, random_state=random_state)
    return TrainResult(
        model=report.model,
        metrics_text=report.metrics_text,
        roc_auc=report.model_metrics.get("roc_auc", float("nan")),
        report=report,
    )
