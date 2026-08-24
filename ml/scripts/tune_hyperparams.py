"""Hyperparameter search for the spike classifier using chronological CV.

Uses sklearn's TimeSeriesSplit (never KFold/shuffled splits) so that every
CV fold trains on strictly earlier data than it validates on -- the same
anti-leakage discipline as the main training pipeline.
"""
import argparse

import joblib
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from src.config import PATHS
from src.features import FeatureConfig
from src.modeling import ALL_FEATURES, TARGET_COL, SplitConfig, build_pipeline, chronological_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-iter", type=int, default=12)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    df = pd.read_csv(PATHS.PROCESSED)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Hold out the most recent slice as an untouched test set; tune only on
    # the earlier train+val portion via chronological CV.
    split = chronological_split(df, SplitConfig(train_frac=0.6, val_frac=0.2))
    tune_df = pd.concat([split.train, split.val], axis=0).sort_values("timestamp")

    X_tune = tune_df[ALL_FEATURES]
    y_tune = tune_df[TARGET_COL].astype(int)

    base = build_pipeline(random_state=args.random_state)

    param_dist = {
        "clf__n_estimators": [200, 400, 800],
        "clf__max_depth": [None, 6, 10, 16],
        "clf__min_samples_split": [2, 4, 8],
        "clf__min_samples_leaf": [1, 2, 4],
    }

    tscv = TimeSeriesSplit(n_splits=args.n_splits)
    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=args.n_iter,
        cv=tscv,
        scoring="roc_auc",
        random_state=args.random_state,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_tune, y_tune)
    best = search.best_estimator_

    # Final honest check on the untouched chronological test set.
    X_test = split.test[ALL_FEATURES]
    y_test = split.test[TARGET_COL].astype(int)
    from sklearn.metrics import roc_auc_score

    test_auc = float("nan")
    if y_test.nunique() > 1:
        test_proba = best.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, test_proba)

    PATHS.MODELS.mkdir(parents=True, exist_ok=True)
    out = PATHS.MODELS / "spike_classifier_tuned.joblib"
    joblib.dump(best, out)

    print("Best params:", search.best_params_)
    print("Best chronological-CV ROC-AUC (train+val):", search.best_score_)
    print("Held-out chronological test ROC-AUC:", test_auc)
    print("Saved tuned model to:", out)
