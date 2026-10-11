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

export type TrendClassification = 'falling' | 'stable' | 'more-variable' | 'rising' | 'not-enough-data';

export interface TrendResult {
  classification: TrendClassification;
  /** Number of distinct local days with at least one reading, used as the trend's sample size. */
  sampleSize: number;
  /** Theil-Sen slope, mg/dL per calendar day (gaps between logged days count). Null when there is not enough data. */
  slopeMgdlPerDay: number | null;
  mannKendallTau: number | null;
  mannKendallPValue: number | null;
  /** Day-to-day variability trend slope (rolling SD), used to distinguish "rising" from "more variable". */
  variabilitySlope: number | null;
}

const DAY_MS = 24 * 60 * 60 * 1000;

/**
 * Whole days since the epoch for a local `YYYY-MM-DD` day key. The slope is
 * fitted against this rather than the index of the day in the list: indexing
 * only days that have readings squeezed a 3-week logging gap down to one step,
 * so a user who logged sporadically got a slope several times steeper than the
 * real per-day change (and "mg/dL per day" was not per calendar day at all).
 */
function calendarDayNumber(dayKey: string): number {
  const [y, m, d] = dayKey.split('-').map(Number);
  return Math.round(Date.UTC(y ?? 1970, (m ?? 1) - 1, d ?? 1) / DAY_MS);
}

function readingsByDay(input: AnalyticsInput): Map<string, number[]> {
  const byDay = new Map<string, number[]>();
  for (const g of input.glucose) {
    const key = dayKeyInZone(g.takenAt, input.timezone);
    const arr = byDay.get(key) ?? [];
    arr.push(g.valueMgdl);
    byDay.set(key, arr);
  }
  return byDay;
}

/** One value per local calendar day: the mean glucose that day, in chronological order. */
function dailyAverages(input: AnalyticsInput): { dayNumber: number[]; values: number[] } {
  const byDay = readingsByDay(input);
  const values: number[] = [];
  const dayNumber: number[] = [];
  for (const key of [...byDay.keys()].sort()) {
    const avg = mean(byDay.get(key) ?? []);
    if (avg !== null && avg !== undefined) {
      dayNumber.push(calendarDayNumber(key));
      values.push(avg);
    }
  }
  return { dayNumber, values };
}

/** Rolling standard deviation isn't needed day-by-day here; instead we track each day's within-day SD as a variability series. */
function dailyVariability(input: AnalyticsInput): { dayNumber: number[]; values: number[] } {
  const byDay = readingsByDay(input);
  const dayNumber: number[] = [];
  const values: number[] = [];
  for (const key of [...byDay.keys()].sort()) {
    const dayValues = byDay.get(key) ?? [];
    if (dayValues.length < 2) continue; // need at least 2 readings to have within-day spread
    const m = mean(dayValues);
    if (m === null) continue;
    const variance = dayValues.reduce((acc, v) => acc + (v - m) ** 2, 0) / (dayValues.length - 1);
    dayNumber.push(calendarDayNumber(key));
    values.push(Math.sqrt(variance));
  }
  return { dayNumber, values };
}

/**
 * Classifies the trend across the whole (already-windowed) input:
 *  - "not-enough-data" below EVIDENCE_THRESHOLDS.trend.early distinct days.
 *  - "rising" when the mean-glucose slope is significantly positive (p < 0.05, Mann-Kendall).
 *  - "falling" when the mean-glucose slope is significantly negative. The label
 *    is deliberately direction-only: a falling average is "improving" for
 *    someone running high and alarming for someone with frequent lows, so the
 *    value judgement is left to the reader (and never sent to the AI layer).
 *  - "more-variable" when the mean slope is flat but day-to-day variability is significantly rising.
 *  - "stable" otherwise.
 */
export function detectTrend(input: AnalyticsInput): TrendResult {
  const { dayNumber, values } = dailyAverages(input);
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

  const ts = theilSen(dayNumber, values);
  const mk = mannKendall(values);

  const { dayNumber: variabilityDays, values: variabilityValues } = dailyVariability(input);
  let variabilitySlope: number | null = null;
  let variabilityMk: ReturnType<typeof mannKendall> = null;
  if (variabilityValues.length >= EVIDENCE_THRESHOLDS.trend!.early) {
    const vts = theilSen(variabilityDays, variabilityValues);
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
    classification = 'falling';
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
