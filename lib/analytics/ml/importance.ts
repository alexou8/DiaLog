/**
 * Personalised feature importance: which logged factors are most
 * associated with this user's post-meal glucose, in their own data.
 *
 * Implementation: standardised (z-scored) ridge regression via the normal
 * equations, beta = (X^T X + lambda I)^-1 X^T y, with X and y both
 * z-scored first. Because every column is standardised, the resulting
 * coefficients are directly comparable in magnitude ("standardised
 * coefficients") and no intercept term is needed (both X and y are
 * centred at 0 by construction).
 *
 * Ridge (rather than plain OLS) is used because the engineered features
 * are frequently correlated with each other (e.g. minutes-since-meal and
 * minutes-since-medication often move together) and a single user's
 * history is a small, noisy sample — plain OLS is unstable in exactly
 * this regime, ridge trades a small amount of bias for much lower
 * variance in the coefficient estimates.
 *
 * This is explicitly NOT a causal model. It reports which of the user's
 * own logged behaviours *move together with* their post-meal glucose,
 * nothing more, and every result carries that caveat.
 */
import { EVIDENCE_THRESHOLDS } from '@/lib/domain/evidence';
import { POST_MEAL_WINDOW_MIN } from '@/lib/domain/thresholds';
import { mean, stdDev } from '../stats';
import { computeGlucoseFeatures, type GlucoseFeatureRow } from '../features';
import type { AnalyticsInput } from '../types';

const CANDIDATE_PREDICTORS = [
  'lastMealCarbsG',
  'minutesSinceMedication',
  'minutesSinceExercise',
  'exerciseMinutesPrior3h',
  'sleepHoursPriorNight',
  'hour',
  'weekday',
] as const;
type PredictorKey = (typeof CANDIDATE_PREDICTORS)[number];

const PREDICTOR_LABELS: Record<PredictorKey, string> = {
  lastMealCarbsG: 'carbs in the meal',
  minutesSinceMedication: 'time since your last logged medication',
  minutesSinceExercise: 'time since your last logged exercise',
  exerciseMinutesPrior3h: 'exercise minutes in the 3h before',
  sleepHoursPriorNight: 'hours slept the night before',
  hour: 'time of day',
  weekday: 'day of the week',
};

export interface FeatureCoefficient {
  feature: PredictorKey;
  label: string;
  standardizedCoefficient: number;
}

export interface FeatureImportanceResult {
  outcome: 'post-meal glucose (60-180 min after a meal)';
  sampleSize: number;
  featuresUsed: PredictorKey[];
  featuresDropped: { feature: PredictorKey; reason: string }[];
  coefficients: FeatureCoefficient[]; // ranked by |standardizedCoefficient| descending
  ridgeLambda: number;
  r2: number | null;
  warning: string;
}

function predictorValue(row: GlucoseFeatureRow, key: PredictorKey): number | null {
  return row[key];
}

// --- minimal dense linear algebra (small matrices only: a handful of features) ---

function transpose(m: number[][]): number[][] {
  const rows = m.length;
  const cols = m[0]?.length ?? 0;
  const out: number[][] = Array.from({ length: cols }, () => new Array(rows).fill(0));
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      out[j]![i] = m[i]![j]!;
    }
  }
  return out;
}

function matMul(a: number[][], b: number[][]): number[][] {
  const n = a.length;
  const k = a[0]?.length ?? 0;
  const m = b[0]?.length ?? 0;
  const out: number[][] = Array.from({ length: n }, () => new Array(m).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      let s = 0;
      for (let t = 0; t < k; t++) s += (a[i]![t] ?? 0) * (b[t]![j] ?? 0);
      out[i]![j] = s;
    }
  }
  return out;
}

/** Gauss-Jordan inversion of a small square matrix. Returns null if singular. */
function invert(matrix: readonly number[][]): number[][] | null {
  const n = matrix.length;
  const a = matrix.map((row) => [...row]);
  const inv: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)),
  );

  for (let col = 0; col < n; col++) {
    let pivotRow = col;
    let maxAbs = Math.abs(a[col]?.[col] ?? 0);
    for (let r = col + 1; r < n; r++) {
      const v = Math.abs(a[r]?.[col] ?? 0);
      if (v > maxAbs) {
        maxAbs = v;
        pivotRow = r;
      }
    }
    if (maxAbs < 1e-12) return null; // singular (shouldn't happen with ridge regularisation)
    if (pivotRow !== col) {
      const tmpA = a[col]!;
      a[col] = a[pivotRow]!;
      a[pivotRow] = tmpA;
      const tmpI = inv[col]!;
      inv[col] = inv[pivotRow]!;
      inv[pivotRow] = tmpI;
    }
    const pivot = a[col]![col]!;
    for (let j = 0; j < n; j++) {
      a[col]![j] = a[col]![j]! / pivot;
      inv[col]![j] = inv[col]![j]! / pivot;
    }
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = a[r]![col]!;
      if (factor === 0) continue;
      for (let j = 0; j < n; j++) {
        a[r]![j] = a[r]![j]! - factor * a[col]![j]!;
        inv[r]![j] = inv[r]![j]! - factor * inv[col]![j]!;
      }
    }
  }
  return inv;
}

function standardizeColumn(values: readonly number[]): { z: number[]; mean: number; sd: number } {
  const m = mean(values) ?? 0;
  const sd = stdDev(values) || 1; // avoid divide-by-zero for a constant column
  return { z: values.map((v) => (v - m) / sd), mean: m, sd };
}

export interface FeatureImportanceOptions {
  ridgeLambda?: number;
  /** Minimum fraction of outcome rows a candidate predictor must have a non-null value for, to be included. */
  minCoverage?: number;
}

/**
 * Refuses (returns null) below EVIDENCE_THRESHOLDS.model.early usable rows
 * — this is a model fit, and a personalised model from a handful of
 * observations is not trustworthy even as a hint.
 */
export function computeFeatureImportance(
  input: AnalyticsInput,
  options: FeatureImportanceOptions = {},
): FeatureImportanceResult | null {
  const ridgeLambda = options.ridgeLambda ?? 1.0;
  const minCoverage = options.minCoverage ?? 0.5;

  const features = computeGlucoseFeatures(input.glucose, input.meals, input.medications, input.exercise, input.sleep, {
    timezone: input.timezone,
  });

  const postMealRows = features.filter(
    (f) => f.minutesSinceMeal !== null && f.minutesSinceMeal >= POST_MEAL_WINDOW_MIN.start && f.minutesSinceMeal <= POST_MEAL_WINDOW_MIN.end,
  );

  if (postMealRows.length === 0) return null;

  const featuresDropped: { feature: PredictorKey; reason: string }[] = [];
  const featuresUsed: PredictorKey[] = [];
  for (const key of CANDIDATE_PREDICTORS) {
    const nonNull = postMealRows.filter((r) => predictorValue(r, key) !== null).length;
    const coverage = nonNull / postMealRows.length;
    if (coverage < minCoverage) {
      featuresDropped.push({ feature: key, reason: `only ${Math.round(coverage * 100)}% of post-meal readings had this logged (need ${Math.round(minCoverage * 100)}%)` });
    } else {
      featuresUsed.push(key);
    }
  }

  if (featuresUsed.length === 0) return null;

  const completeRows = postMealRows.filter((r) => featuresUsed.every((key) => predictorValue(r, key) !== null));

  if (completeRows.length < EVIDENCE_THRESHOLDS.model!.early) return null;

  const y = completeRows.map((r) => r.valueMgdl);
  const rawX = featuresUsed.map((key) => completeRows.map((r) => predictorValue(r, key) as number));

  const yStd = standardizeColumn(y);
  const xStd = rawX.map((col) => standardizeColumn(col));

  // Design matrix: rows = observations, cols = standardised features.
  const X: number[][] = completeRows.map((_, i) => xStd.map((col) => col.z[i] ?? 0));
  const Y: number[][] = yStd.z.map((v) => [v]);

  const Xt = transpose(X);
  const XtX = matMul(Xt, X);
  const ridged = XtX.map((row, i) => row.map((v, j) => v + (i === j ? ridgeLambda : 0)));
  const inv = invert(ridged);
  if (!inv) return null;
  const XtY = matMul(Xt, Y);
  const betaMatrix = matMul(inv, XtY);
  const beta = betaMatrix.map((row) => row[0] ?? 0);

  // R^2 on the standardised outcome.
  const predictions = matMul(X, betaMatrix).map((row) => row[0] ?? 0);
  const ssRes = predictions.reduce((acc, pred, i) => acc + ((Y[i]?.[0] ?? 0) - pred) ** 2, 0);
  const ssTot = Y.reduce((acc, row) => acc + (row[0] ?? 0) ** 2, 0); // standardised y has mean 0
  const r2 = ssTot > 0 ? 1 - ssRes / ssTot : null;

  const coefficients: FeatureCoefficient[] = featuresUsed
    .map((key, i) => ({ feature: key, label: PREDICTOR_LABELS[key], standardizedCoefficient: beta[i] ?? 0 }))
    .sort((a, b) => Math.abs(b.standardizedCoefficient) - Math.abs(a.standardizedCoefficient));

  return {
    outcome: 'post-meal glucose (60-180 min after a meal)',
    sampleSize: completeRows.length,
    featuresUsed,
    featuresDropped,
    coefficients,
    ridgeLambda,
    r2,
    warning:
      'These are associations found in your own logged data, not causes. Coefficients can shift as you log more, and this does not account for factors you have not logged.',
  };
}
