/**
 * Pure statistical primitives used across the analytics engine.
 *
 * No I/O, no Prisma. Every function that is undefined for small samples
 * returns `null` rather than `NaN` or `Infinity`, so callers can propagate
 * "not enough data" without special-casing floating point edge cases.
 */

export interface LinearRegressionResult {
  slope: number;
  intercept: number;
  /** Coefficient of determination, in [0, 1] (0 when variance of y is 0). */
  r2: number;
}

export interface TheilSenResult {
  slope: number;
  intercept: number;
}

export interface MannKendallResult {
  /** Kendall's S statistic: sum of signs of all pairwise differences. */
  s: number;
  /** Kendall's tau (S normalised to [-1, 1]). */
  tau: number;
  /** Standard normal z-score of S under the null of no trend. */
  z: number;
  /** Approximate two-sided p-value from the normal approximation. */
  pValue: number;
}

export interface WelchTTestResult {
  t: number;
  /** Welch–Satterthwaite approximate degrees of freedom. */
  df: number;
  /** Approximate two-sided p-value from Student's t distribution. */
  pValue: number;
  mean1: number;
  mean2: number;
  n1: number;
  n2: number;
}

function isFiniteNumber(x: number): boolean {
  return Number.isFinite(x);
}

function cleanArray(xs: readonly number[]): number[] {
  return xs.filter(isFiniteNumber);
}

export function sum(xs: readonly number[]): number {
  let total = 0;
  for (const x of xs) total += x;
  return total;
}

export function mean(xs: readonly number[]): number | null {
  const clean = cleanArray(xs);
  if (clean.length === 0) return null;
  return sum(clean) / clean.length;
}

/**
 * Median via the standard "average the two middle values" convention.
 * Uses a copy + sort; fine for the dataset sizes this engine operates on
 * (a single user's history, at most a few tens of thousands of readings).
 */
export function median(xs: readonly number[]): number | null {
  return quantile(xs, 0.5);
}

/**
 * Linear-interpolation quantile, equivalent to NumPy's default
 * ("linear") method and Excel's PERCENTILE.INC. `p` in [0, 1].
 */
export function quantile(xs: readonly number[], p: number): number | null {
  const clean = cleanArray(xs)
    .slice()
    .sort((a, b) => a - b);
  const n = clean.length;
  if (n === 0) return null;
  if (p <= 0) return clean[0] ?? null;
  if (p >= 1) return clean[n - 1] ?? null;
  const idx = p * (n - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  const loVal = clean[lo];
  const hiVal = clean[hi];
  if (loVal === undefined || hiVal === undefined) return null;
  if (lo === hi) return loVal;
  const frac = idx - lo;
  return loVal + (hiVal - loVal) * frac;
}

/** Sample standard deviation (n-1 denominator). Null below n=2. */
export function stdDev(xs: readonly number[]): number | null {
  const clean = cleanArray(xs);
  const n = clean.length;
  if (n < 2) return null;
  const m = sum(clean) / n;
  const variance = sum(clean.map((x) => (x - m) ** 2)) / (n - 1);
  return Math.sqrt(variance);
}

/** Coefficient of variation (SD / mean), expressed as a fraction (not %). */
export function coefficientOfVariation(xs: readonly number[]): number | null {
  const m = mean(xs);
  const sd = stdDev(xs);
  if (m === null || sd === null || m === 0) return null;
  return sd / m;
}

/** Interquartile range (Q3 - Q1). */
export function iqr(xs: readonly number[]): number | null {
  const clean = cleanArray(xs);
  if (clean.length < 2) return null;
  const q1 = quantile(clean, 0.25);
  const q3 = quantile(clean, 0.75);
  if (q1 === null || q3 === null) return null;
  return q3 - q1;
}

/**
 * Median Absolute Deviation. `scale = 1.4826` (default) makes MAD a
 * consistent estimator of the standard deviation under a normal
 * distribution, which is what modified-z-score anomaly detection assumes.
 * Pass `scale: 1` for the raw MAD.
 */
export function mad(xs: readonly number[], scale = 1.4826): number | null {
  const clean = cleanArray(xs);
  if (clean.length === 0) return null;
  const m = median(clean);
  if (m === null) return null;
  const deviations = clean.map((x) => Math.abs(x - m));
  const rawMad = median(deviations);
  if (rawMad === null) return null;
  return rawMad * scale;
}

/**
 * Modified z-scores (Iglewicz & Hoaglin, 1993): 0.6745 * (x - median) / MAD.
 * The 0.6745 constant makes the modified z-score comparable in magnitude to
 * a standard z-score under normality. Returns null per-element when the
 * baseline MAD is 0 (all baseline values identical) since the score would
 * be undefined or infinite.
 */
export function modifiedZScores(
  xs: readonly number[],
  baseline: readonly number[] = xs,
): (number | null)[] {
  const baseMedian = median(baseline);
  const rawMad = mad(baseline, 1); // unscaled MAD, we apply the 0.6745 constant ourselves
  return xs.map((x) => {
    if (!isFiniteNumber(x) || baseMedian === null || rawMad === null || rawMad === 0) return null;
    return (0.6745 * (x - baseMedian)) / rawMad;
  });
}

/** Pearson product-moment correlation. Null below n=2 or zero variance. */
export function pearson(xs: readonly number[], ys: readonly number[]): number | null {
  const n = Math.min(xs.length, ys.length);
  if (n < 2) return null;
  const pairs: [number, number][] = [];
  for (let i = 0; i < n; i++) {
    const x = xs[i];
    const y = ys[i];
    if (x !== undefined && y !== undefined && isFiniteNumber(x) && isFiniteNumber(y))
      pairs.push([x, y]);
  }
  if (pairs.length < 2) return null;
  const mx = sum(pairs.map((p) => p[0])) / pairs.length;
  const my = sum(pairs.map((p) => p[1])) / pairs.length;
  let cov = 0;
  let vx = 0;
  let vy = 0;
  for (const [x, y] of pairs) {
    const dx = x - mx;
    const dy = y - my;
    cov += dx * dy;
    vx += dx * dx;
    vy += dy * dy;
  }
  if (vx === 0 || vy === 0) return null;
  return cov / Math.sqrt(vx * vy);
}

/** Ranks with average-rank tie handling (needed for Spearman). */
function rankOf(xs: readonly number[]): number[] {
  const n = xs.length;
  const indexed = xs.map((v, i) => ({ v, i }));
  indexed.sort((a, b) => a.v - b.v);
  const ranks = new Array<number>(n);
  let i = 0;
  while (i < n) {
    let j = i;
    while (j + 1 < n && indexed[j + 1]!.v === indexed[i]!.v) j++;
    const avgRank = (i + j) / 2 + 1; // 1-based average rank across the tie block
    for (let k = i; k <= j; k++) {
      const entry = indexed[k];
      if (entry) ranks[entry.i] = avgRank;
    }
    i = j + 1;
  }
  return ranks;
}

/** Spearman rank correlation (Pearson correlation of the ranks). */
export function spearman(xs: readonly number[], ys: readonly number[]): number | null {
  const n = Math.min(xs.length, ys.length);
  if (n < 2) return null;
  const cleanXs: number[] = [];
  const cleanYs: number[] = [];
  for (let i = 0; i < n; i++) {
    const x = xs[i];
    const y = ys[i];
    if (x !== undefined && y !== undefined && isFiniteNumber(x) && isFiniteNumber(y)) {
      cleanXs.push(x);
      cleanYs.push(y);
    }
  }
  if (cleanXs.length < 2) return null;
  return pearson(rankOf(cleanXs), rankOf(cleanYs));
}

/** Ordinary least squares simple linear regression, y = slope*x + intercept. */
export function linearRegression(
  xs: readonly number[],
  ys: readonly number[],
): LinearRegressionResult | null {
  const n = Math.min(xs.length, ys.length);
  if (n < 2) return null;
  const pairs: [number, number][] = [];
  for (let i = 0; i < n; i++) {
    const x = xs[i];
    const y = ys[i];
    if (x !== undefined && y !== undefined && isFiniteNumber(x) && isFiniteNumber(y))
      pairs.push([x, y]);
  }
  if (pairs.length < 2) return null;
  const mx = sum(pairs.map((p) => p[0])) / pairs.length;
  const my = sum(pairs.map((p) => p[1])) / pairs.length;
  let sxy = 0;
  let sxx = 0;
  let syy = 0;
  for (const [x, y] of pairs) {
    sxy += (x - mx) * (y - my);
    sxx += (x - mx) ** 2;
    syy += (y - my) ** 2;
  }
  if (sxx === 0) return null;
  const slope = sxy / sxx;
  const intercept = my - slope * mx;
  const r2 = syy === 0 ? 1 : (sxy * sxy) / (sxx * syy);
  return { slope, intercept, r2 };
}

/**
 * Theil–Sen estimator: the median of all pairwise slopes. Robust to
 * outliers (breakdown point ~29%) which matters for glucose data where a
 * single unusual day should not dominate a trend line. O(n^2) pairwise
 * slopes — fine for rolling windows of a few hundred points at most.
 */
export function theilSen(xs: readonly number[], ys: readonly number[]): TheilSenResult | null {
  const n = Math.min(xs.length, ys.length);
  if (n < 2) return null;
  const slopes: number[] = [];
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const xi = xs[i];
      const xj = xs[j];
      const yi = ys[i];
      const yj = ys[j];
      if (xi === undefined || xj === undefined || yi === undefined || yj === undefined) continue;
      if (!isFiniteNumber(xi) || !isFiniteNumber(xj) || !isFiniteNumber(yi) || !isFiniteNumber(yj))
        continue;
      const dx = xj - xi;
      if (dx === 0) continue;
      slopes.push((yj - yi) / dx);
    }
  }
  if (slopes.length === 0) return null;
  const slope = median(slopes);
  if (slope === null) return null;
  const intercepts: number[] = [];
  for (let i = 0; i < n; i++) {
    const xi = xs[i];
    const yi = ys[i];
    if (xi === undefined || yi === undefined || !isFiniteNumber(xi) || !isFiniteNumber(yi))
      continue;
    intercepts.push(yi - slope * xi);
  }
  const intercept = median(intercepts);
  if (intercept === null) return null;
  return { slope, intercept };
}

/** Standard normal cumulative distribution function via Abramowitz & Stegun 7.1.26 (max error ~1.5e-7). */
function normalCdf(z: number): number {
  const sign = z < 0 ? -1 : 1;
  const x = Math.abs(z) / Math.SQRT2;
  const t = 1 / (1 + 0.3275911 * x);
  const y =
    1 -
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) *
      t *
      Math.exp(-x * x);
  return 0.5 * (1 + sign * y);
}

/**
 * Mann–Kendall trend test. Tests the null hypothesis that there is no
 * monotonic trend in `ys` (assumed to be in time order).
 *
 * The p-value uses the standard normal approximation to the distribution
 * of S with a continuity correction and a tie correction in the variance
 * term. This approximation is the textbook one (Gilbert 1987) and is
 * accurate for n >= ~10; for smaller n the exact distribution differs
 * somewhat, so callers should treat p-values from very small samples as
 * indicative rather than exact (we additionally gate this analysis behind
 * EVIDENCE_THRESHOLDS.trend at the call site).
 */
export function mannKendall(ys: readonly number[]): MannKendallResult | null {
  const clean = cleanArray(ys);
  const n = clean.length;
  if (n < 2) return null;

  let s = 0;
  for (let i = 0; i < n - 1; i++) {
    for (let j = i + 1; j < n; j++) {
      const a = clean[i];
      const b = clean[j];
      if (a === undefined || b === undefined) continue;
      s += Math.sign(b - a);
    }
  }

  // Tie correction: group equal values and subtract their contribution to variance.
  const sorted = [...clean].sort((a, b) => a - b);
  let tieTermSum = 0;
  let i = 0;
  while (i < n) {
    let j = i;
    while (j + 1 < n && sorted[j + 1] === sorted[i]) j++;
    const tieSize = j - i + 1;
    if (tieSize > 1) tieTermSum += tieSize * (tieSize - 1) * (2 * tieSize + 5);
    i = j + 1;
  }
  const variance = (n * (n - 1) * (2 * n + 5) - tieTermSum) / 18;
  if (variance <= 0) return { s, tau: 0, z: 0, pValue: 1 };

  const z = s > 0 ? (s - 1) / Math.sqrt(variance) : s < 0 ? (s + 1) / Math.sqrt(variance) : 0;
  const pValue = 2 * (1 - normalCdf(Math.abs(z)));

  const totalPairs = (n * (n - 1)) / 2;
  const tau = totalPairs === 0 ? 0 : s / totalPairs;

  return { s, tau, z, pValue: Math.min(1, Math.max(0, pValue)) };
}

/**
 * Regularized incomplete beta function I_x(a, b) via the continued-fraction
 * expansion (Numerical Recipes, "betacf" + "betai"). This is the standard
 * numerically stable way to evaluate the Student's t CDF; double-precision
 * accuracy is typically better than 1e-10 for the parameter ranges we use
 * here (a, b representing degrees of freedom in the 1-1e6 range).
 */
function logGamma(x: number): number {
  // Lanczos approximation, g=7, n=9 coefficients. Accurate to ~15 significant digits.
  const g = 7;
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6,
    1.5056327351493116e-7,
  ];
  if (x < 0.5) {
    return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
  }
  const xx = x - 1;
  let a = c[0]!;
  const t = xx + g + 0.5;
  for (let i = 1; i < g + 2; i++) {
    a += c[i]! / (xx + i);
  }
  return 0.5 * Math.log(2 * Math.PI) + (xx + 0.5) * Math.log(t) - t + Math.log(a);
}

function betacf(x: number, a: number, b: number): number {
  const MAXIT = 200;
  const EPS = 3e-14;
  const FPMIN = 1e-300;
  const qab = a + b;
  const qap = a + 1;
  const qam = a - 1;
  let c = 1;
  let d = 1 - (qab * x) / qap;
  if (Math.abs(d) < FPMIN) d = FPMIN;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= MAXIT; m++) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    h *= d * c;
    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < EPS) break;
  }
  return h;
}

function regularizedIncompleteBeta(x: number, a: number, b: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const lbeta = logGamma(a) + logGamma(b) - logGamma(a + b);
  const front = Math.exp(Math.log(x) * a + Math.log(1 - x) * b - lbeta);
  if (x < (a + 1) / (a + b + 2)) {
    return (front * betacf(x, a, b)) / a;
  }
  return 1 - (front * betacf(1 - x, b, a)) / b;
}

/** Two-sided p-value for Student's t distribution with `df` degrees of freedom. */
export function tDistributionTwoSidedPValue(t: number, df: number): number {
  if (!isFiniteNumber(t) || !isFiniteNumber(df) || df <= 0) return 1;
  const x = df / (df + t * t);
  const p = regularizedIncompleteBeta(x, df / 2, 0.5);
  return Math.min(1, Math.max(0, p));
}

/**
 * Welch's t-test for two independent samples with possibly unequal
 * variances. Returns null below n=2 in either group.
 */
export function welchTTest(xs: readonly number[], ys: readonly number[]): WelchTTestResult | null {
  const x = cleanArray(xs);
  const y = cleanArray(ys);
  const n1 = x.length;
  const n2 = y.length;
  if (n1 < 2 || n2 < 2) return null;
  const m1 = mean(x);
  const m2 = mean(y);
  const sd1 = stdDev(x);
  const sd2 = stdDev(y);
  if (m1 === null || m2 === null || sd1 === null || sd2 === null) return null;
  const v1 = sd1 * sd1;
  const v2 = sd2 * sd2;
  const se2 = v1 / n1 + v2 / n2;
  if (se2 === 0) return null;
  const se = Math.sqrt(se2);
  const t = (m1 - m2) / se;
  const df = (se2 * se2) / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1));
  const pValue = tDistributionTwoSidedPValue(t, df);
  return { t, df, pValue, mean1: m1, mean2: m2, n1, n2 };
}

/**
 * Cohen's d using the pooled standard deviation of the two samples
 * (the standard "equal weight" pooling, not sample-size weighted — this
 * matches the most common textbook definition).
 */
export function cohensD(xs: readonly number[], ys: readonly number[]): number | null {
  const x = cleanArray(xs);
  const y = cleanArray(ys);
  const n1 = x.length;
  const n2 = y.length;
  if (n1 < 2 || n2 < 2) return null;
  const m1 = mean(x);
  const m2 = mean(y);
  const sd1 = stdDev(x);
  const sd2 = stdDev(y);
  if (m1 === null || m2 === null || sd1 === null || sd2 === null) return null;
  const pooled = Math.sqrt(((n1 - 1) * sd1 * sd1 + (n2 - 1) * sd2 * sd2) / (n1 + n2 - 2));
  if (pooled === 0) return null;
  return (m1 - m2) / pooled;
}
