# ML Pipeline (Offline Research)

`ml/` is DiaLog's **offline research / model-development pipeline**. It is a
standalone Python project, separate from the production Next.js/TypeScript
application at the repository root. It is used to prototype and honestly
evaluate statistical methods -- for example, "does recent carb intake plus
medication timing predict a glucose spike?" -- so a human reviewer can decide
whether a method is worth reimplementing in the production TypeScript
analytics engine. **The pipeline itself is never deployed**; it does not run
in the product and its models are not served to users.

## Relationship to the production analytics engine

| | `ml/` (this pipeline) | Production TS analytics engine (repo root) |
|---|---|---|
| Purpose | Research: does a method work at all, and how well? | Serves real users |
| Language | Python (pandas, scikit-learn) | TypeScript |
| Data | Local sqlite export / CSV, including synthetic fixtures | Live user data |
| Deployment | Not deployed, run manually / in CI as a check | Deployed |
| Role | Produces a findings report a human reads | Implements whatever the findings justify |

Nothing in `ml/` is imported by, or executes inside, the production app. The
intended workflow is: prototype and evaluate a method here -> read the
findings report (`ml/scripts/export_findings.py` output) -> if the lift over
baselines is real, sufficiently large, and measured on real (not synthetic)
data, an engineer reimplements the validated method in the TypeScript
analytics engine, independently, with its own tests.

## Leakage issues found and fixed

An audit of the pipeline as originally written found several data-science
integrity problems that would have overstated model performance. All were
fixed in `ml/src/features.py` and `ml/src/modeling.py`:

1. **Random train/test split on time-series data.** The original
   `train_classifier` used `sklearn.model_selection.train_test_split` with
   `stratify`, i.e. a random shuffle. Because the underlying data is a time
   series (glucose readings anchored to timestamps), a random split lets the
   model train on readings that are chronologically *after* some of the
   readings it is evaluated on -- an easy way to look better than a
   real-world forward-predicting deployment ever would. **Fix:** the model
   is now trained and evaluated with a strict chronological split
   (`src.modeling.chronological_split`): the earliest 60% of rows train the
   model, the next 20% are a validation slice, and the most recent 20% are
   the untouched test set. `chronological_split` asserts (not just
   documents) that every train timestamp precedes every test timestamp.
   Hyperparameter tuning (`scripts/tune_hyperparams.py`) uses
   `sklearn.model_selection.TimeSeriesSplit` instead of the default k-fold
   CV, for the same reason -- every CV fold trains on strictly earlier data
   than it validates on.

2. **Median imputation fit on the whole dataset.** The original
   `build_training_examples` called
   `df["mins_since_meal"].fillna(df["mins_since_meal"].median())` on the
   full dataset before any split -- meaning the imputed value for a
   training-set row was influenced by test-set rows' values. **Fix:**
   `build_training_examples` no longer imputes at all; it leaves missing
   values as `NaN`. Imputation now happens inside the model's
   `sklearn.pipeline.Pipeline` (`SimpleImputer`, median for numeric /
   most-frequent for categorical), which is fit only on whatever data is
   passed to `.fit()` -- the training fold, and never the validation or test
   fold. `ml/tests/unit/test_modeling.py::TestImputationFoldDiscipline`
   is a regression test that recomputes the training-fold median directly
   and asserts the fitted imputer matches it (and differs from the
   whole-dataset median where the two diverge).

3. **No baseline comparisons.** The original pipeline reported a raw
   ROC-AUC / classification report with nothing to compare it against, which
   invites reading "0.82 AUC" as impressive in isolation. **Fix:** every
   evaluation now reports three baselines alongside the model, computed with
   the same discipline (fit only on the training fold where applicable):
   - **Majority-class baseline** -- always predict the more common class
     from the training fold.
   - **Prevalence baseline** -- always predict the training fold's positive
     rate as a constant probability.
   - **Previous-reading (persistence) baseline** -- predict that the spike
     label repeats from the immediately preceding chronological reading (the
     "naive forecast" any real monitoring model needs to beat). This uses
     only strictly-past information and is not itself leakage.

   The model's **lift over each baseline** (model metric minus baseline
   metric, per metric) is reported explicitly rather than left for the
   reader to compute.

4. **No uncertainty reporting.** A single point-estimate metric on a test
   set can look very different from run to run, especially on small
   samples. **Fix:** every headline metric ships with a bootstrap 95%
   confidence interval (`src.modeling.bootstrap_metric_ci`, 1000 resamples
   of the test set by default), plus the raw test-set sample size and
   per-split class balance. When the test set has fewer than 30 rows, the
   report includes an explicit warning that the metrics and CIs are
   unreliable at that sample size, rather than presenting a false sense of
   precision.

5. **Magic numbers for the spike threshold and lookback window.** The
   200 mg/dL spike threshold and 6-hour meal/med lookback window were
   hardcoded in multiple places. **Fix:** both are now fields on
   `src.features.FeatureConfig`, threaded through
   `build_training_examples`, and exposed as CLI flags
   (`--spike-threshold`, `--lookback-hours`) on `scripts/preprocess_data.py`,
   `scripts/train_model.py`, and `scripts/export_findings.py`.

## Evaluation methodology (current)

For a given labeled event log:

1. Build features (`src.features.build_training_examples`) anchored to each
   glucose reading, using only events at or before that reading's
   timestamp. Each row also carries `prev_label_spike`, the label of the
   immediately preceding reading, for the persistence baseline.
2. Split chronologically into train (60%) / validation (20%) / test (20%)
   (`src.modeling.chronological_split`).
3. Fit a `RandomForestClassifier` inside a `Pipeline` whose imputation and
   one-hot encoding are fit on the training fold only.
4. Evaluate on the untouched test fold: accuracy, precision, recall, F1,
   ROC-AUC, and Brier score, each with a bootstrap 95% CI where defined.
5. Compute the same metrics for the majority-class, prevalence, and
   previous-reading baselines on the same test fold, and the model's lift
   over each.
6. Additionally run `sklearn.model_selection.TimeSeriesSplit` cross-validation
   over the train+validation portion, for a robustness check beyond the
   single held-out test slice.
7. Report sample sizes and class balance (positive rate) for every split.

All of this is emitted as a single JSON document by
`ml/scripts/export_findings.py`, plus a `feature_importances` field and an
explicit `data_provenance` field (see below) -- this is what a reviewer
reads before deciding a method is worth porting to production.

## Data provenance and synthetic data

`ml/data/sample_glucose_data.csv` is generated by
`ml/examples/generate_sample_data.py`: a smooth sinusoidal daily glucose
curve, synthetic meal spikes at fixed hours, and Gaussian noise --
parameters chosen for a runnable demo, not fit to or validated against real
physiology. `ml/data/sample_logs.csv` is a small hand-written synthetic
event log used for pipeline smoke tests.

**Any metric produced from either file, or from any other script-generated
data, carries no clinical validity.** This is not only stated in this
document: `src.modeling.train_and_evaluate` requires callers to pass a
`data_provenance` string (there is no silent default that looks like real
data), and `scripts/export_findings.py` auto-labels any events file whose
name contains `sample`, `synthetic`, `generated`, or `fixture` as
`SYNTHETIC ... -- no clinical validity` in the `data_provenance` field of
its JSON output, unless the caller overrides it with an explicit
`--data-provenance` string. `data_provenance` is a required, non-optional
field in the exported findings JSON, so it cannot be silently dropped when
a report is read or forwarded.

**Bottom line: nothing computed from `ml/data/sample_glucose_data.csv`,
`ml/data/sample_logs.csv`, or any other script-generated fixture should be
read as evidence about how a method would perform on real patients.** These
datasets exist only to exercise the pipeline's plumbing (splitting,
training, baseline computation, reporting) end-to-end.

## Honest headline numbers (synthetic demo data only)

As a concrete illustration of the methodology above -- **not** a claim about
real-world performance -- running the pipeline on a larger locally-generated
synthetic event log (480 events / 60 simulated days, glucose correlated
with recent carb intake plus noise) produced:

- Sample sizes: 107 train / 35 validation / 37 test rows.
- Test-set class balance: 8.1% positive (spike) rate.
- Model ROC-AUC on the chronological test set: **0.824** (bootstrap 95% CI
  approximately [0.62, 1.00] -- wide, because the test set is only 37 rows).
- Baseline ROC-AUC on the same test set: majority-class 0.500, prevalence
  0.500, previous-reading persistence 0.441.
- Model lift over baselines: **+0.32** ROC-AUC over majority-class/prevalence,
  **+0.38** over the previous-reading baseline.
- Precision/recall on the minority (spike) class was 0.00/0.00 in that run
  (3 positive test examples; the model did not flag any of them) despite the
  favorable ROC-AUC -- a reminder that ROC-AUC alone can look encouraging
  while the model is still useless for the class that actually matters,
  especially at this sample size.

These numbers come entirely from synthetic, script-generated data and exist
only to demonstrate that the reporting pipeline surfaces exactly this kind
of nuance (a promising ranking metric alongside a failing precision/recall
on the class of interest) rather than only the flattering number. They say
nothing about real glucose-spike predictability and must not be cited as
such.
