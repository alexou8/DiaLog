/**
 * Trend detection over a rolling window of daily glucose averages.
 *
 * Combines two robust, non-parametric methods rather than plain OLS on
 * raw readings: Theil-Sen for a slope estimate that isn't dragged around
 * by a handful of outlier days, and Mann-Kendall for a significance test
 * of monotonic trend that makes no distributional assumptions. This pair
 * is standard practice for environmental/health time series specifically
 * because both are robust to the kind of noisy, unevenly-spaced data a
 * self-tracked glucose log produces.
 */
import { dayKeyInZone } from '@/lib/domain/time';
import { mannKendall, mean, theilSen } from '../stats';
import type { AnalyticsInput } from '../types';
import { EVIDENCE_THRESHOLDS } from '@/lib/domain/evidence';

export type TrendClassification = 'improving' | 'stable' | 'more-variable' | 'rising' | 'not-enough-data';

export interface TrendResult {
  classification: TrendClassification;
  /** Number of distinct local days with at least one reading, used as the trend's sample size. */
  sampleSize: number;
  /** Theil-Sen slope, mg/dL per day. Null when there is not enough data. */
  slopeMgdlPerDay: number | null;
  mannKendallTau: number | null;
  mannKendallPValue: number | null;
  /** Day-to-day variability trend slope (rolling SD), used to distinguish "rising" from "more variable". */
  variabilitySlope: number | null;
}

/** One value per local calendar day: the mean glucose that day, in chronological order. */
function dailyAverages(input: AnalyticsInput): { dayIndex: number[]; values: number[] } {
  const byDay = new Map<string, number[]>();
  for (const g of input.glucose) {
    const key = dayKeyInZone(g.takenAt, input.timezone);
    const arr = byDay.get(key) ?? [];
    arr.push(g.valueMgdl);
    byDay.set(key, arr);
  }
  const sortedKeys = [...byDay.keys()].sort();
  const values: number[] = [];
  const dayIndex: number[] = [];
  sortedKeys.forEach((key, i) => {
    const dayValues = byDay.get(key);
    const avg = dayValues ? mean(dayValues) : null;
    if (avg !== null && avg !== undefined) {
      dayIndex.push(i);
      values.push(avg);
    }
  });
  return { dayIndex, values };
}

/** Rolling standard deviation isn't needed day-by-day here; instead we track each day's within-day SD as a variability series. */
function dailyVariability(input: AnalyticsInput): { dayIndex: number[]; values: number[] } {
  const byDay = new Map<string, number[]>();
  for (const g of input.glucose) {
    const key = dayKeyInZone(g.takenAt, input.timezone);
    const arr = byDay.get(key) ?? [];
    arr.push(g.valueMgdl);
    byDay.set(key, arr);
  }
  const sortedKeys = [...byDay.keys()].sort();
  const dayIndex: number[] = [];
  const values: number[] = [];
  sortedKeys.forEach((key, i) => {
    const dayValues = byDay.get(key) ?? [];
    if (dayValues.length < 2) return; // need at least 2 readings to have within-day spread
    const m = mean(dayValues);
    if (m === null) return;
    const variance = dayValues.reduce((acc, v) => acc + (v - m) ** 2, 0) / (dayValues.length - 1);
    dayIndex.push(i);
    values.push(Math.sqrt(variance));
  });
  return { dayIndex, values };
}

/**
 * Classifies the trend across the whole (already-windowed) input:
 *  - "not-enough-data" below EVIDENCE_THRESHOLDS.trend.early distinct days.
 *  - "rising" when the mean-glucose slope is significantly positive (p < 0.05, Mann-Kendall).
 *  - "improving" when the mean-glucose slope is significantly negative.
 *  - "more-variable" when the mean slope is flat but day-to-day variability is significantly rising.
 *  - "stable" otherwise.
 */
export function detectTrend(input: AnalyticsInput): TrendResult {
  const { values } = dailyAverages(input);
  const sampleSize = values.length;

  if (sampleSize < EVIDENCE_THRESHOLDS.trend!.early) {
    return {
      classification: 'not-enough-data',
      sampleSize,
      slopeMgdlPerDay: null,
      mannKendallTau: null,
      mannKendallPValue: null,
      variabilitySlope: null,
    };
  }

  const xs = values.map((_, i) => i);
  const ts = theilSen(xs, values);
  const mk = mannKendall(values);

  const { values: variabilityValues } = dailyVariability(input);
  let variabilitySlope: number | null = null;
  let variabilityMk: ReturnType<typeof mannKendall> = null;
  if (variabilityValues.length >= EVIDENCE_THRESHOLDS.trend!.early) {
    const vxs = variabilityValues.map((_, i) => i);
    const vts = theilSen(vxs, variabilityValues);
    variabilitySlope = vts?.slope ?? null;
    variabilityMk = mannKendall(variabilityValues);
  }

  const SIGNIFICANCE = 0.05;
  const meanSlopeSignificant = mk !== null && mk.pValue < SIGNIFICANCE;
  const variabilitySignificant = variabilityMk !== null && variabilityMk.pValue < SIGNIFICANCE && (variabilitySlope ?? 0) > 0;

  let classification: TrendClassification;
  if (meanSlopeSignificant && (ts?.slope ?? 0) > 0) {
    classification = 'rising';
  } else if (meanSlopeSignificant && (ts?.slope ?? 0) < 0) {
    classification = 'improving';
  } else if (variabilitySignificant) {
    classification = 'more-variable';
  } else {
    classification = 'stable';
  }

  return {
    classification,
    sampleSize,
    slopeMgdlPerDay: ts?.slope ?? null,
    mannKendallTau: mk?.tau ?? null,
    mannKendallPValue: mk?.pValue ?? null,
    variabilitySlope,
  };
}
