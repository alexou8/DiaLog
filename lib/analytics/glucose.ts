/**
 * Deterministic (non-ML) summaries over a set of glucose readings.
 *
 * IMPORTANT — these percentages are "percentage of *readings*" in each
 * band, not clinical CGM "time in range" (TIR). TIR is computed from a
 * continuous stream of sensor readings taken every few minutes and
 * genuinely represents time; fingerstick readings are sparse and
 * irregularly spaced, so counting readings systematically over- or
 * under-represents time depending on when the user chose to test (e.g.
 * testing mostly before meals biases toward pre-meal values). Every
 * consumer-facing label built from this module must say "of your
 * readings", never "of the time".
 */
import {
  dayKeyInZone,
  hourInZone,
  minutesBetween,
  timeOfDayBucket,
  weekdayInZone,
} from '@/lib/domain/time';
import type { TargetRange } from '@/lib/domain/thresholds';
import type { GlucoseContext } from '@prisma/client';
import type { GlucosePoint } from './types';
import {
  cohensD,
  coefficientOfVariation,
  mean,
  median,
  quantile,
  stdDev,
  welchTTest,
} from './stats';
import type { WelchTTestResult } from './stats';

export interface HistogramBucket {
  /** Inclusive lower bound, mg/dL. */
  from: number;
  /** Exclusive upper bound, mg/dL (Infinity for the overflow bucket). */
  to: number;
  count: number;
  fraction: number;
}

export interface HourOfDayProfile {
  hour: number; // 0-23, local time
  count: number;
  averageMgdl: number | null;
  medianMgdl: number | null;
}

export interface WeekdayProfile {
  weekday: number; // 0=Sunday .. 6=Saturday, local time
  count: number;
  averageMgdl: number | null;
  medianMgdl: number | null;
}

export interface ContextProfile {
  context: GlucoseContext;
  count: number;
  averageMgdl: number | null;
  percentInRange: number | null;
}

export interface GroupComparison {
  groupALabel: string;
  groupBLabel: string;
  nA: number;
  nB: number;
  meanA: number | null;
  meanB: number | null;
  test: WelchTTestResult | null;
  cohensD: number | null;
}

export interface WeekBucketSummary {
  /** ISO date key (local timezone) of the first day of this 7-day block. */
  weekStart: string;
  count: number;
  averageMgdl: number | null;
  sdMgdl: number | null;
}

export interface GlucoseSummary {
  count: number;
  averageMgdl: number | null;
  medianMgdl: number | null;
  sdMgdl: number | null;
  /** SD / mean, a unitless measure of glycaemic variability. */
  cv: number | null;
  percentInRange: number | null;
  percentBelowRange: number | null;
  percentAboveRange: number | null;
  /** Readings / (span of the period in days), not calendar days actually logged. */
  readingsPerDay: number | null;
  /** Distinct local calendar days that have at least one reading. */
  daysWithReadings: number;
  averageGapMinutes: number | null;
  medianGapMinutes: number | null;
  maxGapMinutes: number | null;
  histogram: HistogramBucket[];
  byHourOfDay: HourOfDayProfile[];
  byWeekday: WeekdayProfile[];
  byContext: ContextProfile[];
  morningVsEvening: GroupComparison | null;
  weekOverWeek: WeekBucketSummary[];
}

function valuesOf(points: readonly GlucosePoint[]): number[] {
  return points.map((p) => p.valueMgdl);
}

function classifyBand(value: number, range: TargetRange): 'below' | 'in' | 'above' {
  if (value < range.lowMgdl) return 'below';
  if (value > range.highMgdl) return 'above';
  return 'in';
}

function percentInBand(
  points: readonly GlucosePoint[],
  range: TargetRange,
  band: 'below' | 'in' | 'above',
): number | null {
  if (points.length === 0) return null;
  const matching = points.filter((p) => classifyBand(p.valueMgdl, range) === band).length;
  return (matching / points.length) * 100;
}

/** Fixed-width histogram from `min` (inclusive) to `max` (exclusive), with an overflow bucket for values >= max. */
export function histogram(
  points: readonly GlucosePoint[],
  binWidth = 20,
  min = 40,
  max = 400,
): HistogramBucket[] {
  const buckets: HistogramBucket[] = [];
  for (let from = min; from < max; from += binWidth) {
    buckets.push({ from, to: from + binWidth, count: 0, fraction: 0 });
  }
  buckets.push({ from: max, to: Infinity, count: 0, fraction: 0 });
  const underflow = { from: -Infinity, to: min, count: 0, fraction: 0 };
  const total = points.length;
  for (const p of points) {
    const v = p.valueMgdl;
    if (v < min) {
      underflow.count++;
      continue;
    }
    if (v >= max) {
      buckets[buckets.length - 1]!.count++;
      continue;
    }
    const idx = Math.floor((v - min) / binWidth);
    const bucket = buckets[idx];
    if (bucket) bucket.count++;
  }
  const all = underflow.count > 0 ? [underflow, ...buckets] : buckets;
  for (const b of all) b.fraction = total > 0 ? b.count / total : 0;
  return all;
}

export function byHourOfDay(points: readonly GlucosePoint[], timezone: string): HourOfDayProfile[] {
  const groups = new Map<number, number[]>();
  for (const p of points) {
    const h = hourInZone(p.takenAt, timezone);
    const arr = groups.get(h) ?? [];
    arr.push(p.valueMgdl);
    groups.set(h, arr);
  }
  const out: HourOfDayProfile[] = [];
  for (let h = 0; h < 24; h++) {
    const values = groups.get(h) ?? [];
    out.push({
      hour: h,
      count: values.length,
      averageMgdl: mean(values),
      medianMgdl: median(values),
    });
  }
  return out;
}

export function byWeekday(points: readonly GlucosePoint[], timezone: string): WeekdayProfile[] {
  const groups = new Map<number, number[]>();
  for (const p of points) {
    const w = weekdayInZone(p.takenAt, timezone);
    const arr = groups.get(w) ?? [];
    arr.push(p.valueMgdl);
    groups.set(w, arr);
  }
  const out: WeekdayProfile[] = [];
  for (let w = 0; w < 7; w++) {
    const values = groups.get(w) ?? [];
    out.push({
      weekday: w,
      count: values.length,
      averageMgdl: mean(values),
      medianMgdl: median(values),
    });
  }
  return out;
}

export function byContext(points: readonly GlucosePoint[], range: TargetRange): ContextProfile[] {
  const groups = new Map<GlucoseContext, GlucosePoint[]>();
  for (const p of points) {
    const arr = groups.get(p.context) ?? [];
    arr.push(p);
    groups.set(p.context, arr);
  }
  const out: ContextProfile[] = [];
  for (const [context, pts] of groups) {
    out.push({
      context,
      count: pts.length,
      averageMgdl: mean(valuesOf(pts)),
      percentInRange: percentInBand(pts, range, 'in'),
    });
  }
  return out;
}

/** Gaps (minutes) between consecutive readings, sorted chronologically. */
export function gapsMinutes(points: readonly GlucosePoint[]): number[] {
  const sorted = [...points].sort((a, b) => a.takenAt.getTime() - b.takenAt.getTime());
  const gaps: number[] = [];
  for (let i = 1; i < sorted.length; i++) {
    const prev = sorted[i - 1];
    const cur = sorted[i];
    if (prev && cur) gaps.push(minutesBetween(prev.takenAt, cur.takenAt));
  }
  return gaps;
}

/**
 * Compares "morning" readings (overnight + morning time-of-day buckets,
 * i.e. before noon local time — this captures fasting/overnight glucose)
 * against "evening" readings (the evening bucket, local time, 18:00-24:00).
 * Uses Welch's t-test since the two groups can have very different sizes
 * and variances.
 */
export function compareMorningVsEvening(
  points: readonly GlucosePoint[],
  timezone: string,
): GroupComparison | null {
  const morning: number[] = [];
  const evening: number[] = [];
  for (const p of points) {
    const bucket = timeOfDayBucket(hourInZone(p.takenAt, timezone));
    if (bucket === 'overnight' || bucket === 'morning') morning.push(p.valueMgdl);
    else if (bucket === 'evening') evening.push(p.valueMgdl);
  }
  if (morning.length < 2 || evening.length < 2) return null;
  return {
    groupALabel: 'morning (overnight/fasting)',
    groupBLabel: 'evening',
    nA: morning.length,
    nB: evening.length,
    meanA: mean(morning),
    meanB: mean(evening),
    test: welchTTest(morning, evening),
    cohensD: cohensD(morning, evening),
  };
}

/**
 * Buckets readings into consecutive 7-day windows anchored to `periodStart`
 * (in local calendar days) and summarises each window. This is a simple
 * rolling calendar bucketing, not an ISO week — it is anchored to the
 * requested period rather than to Monday, so "week 1" is always the first
 * seven days of the report.
 */
export function weekOverWeek(
  points: readonly GlucosePoint[],
  timezone: string,
  periodStart: Date,
): WeekBucketSummary[] {
  if (points.length === 0) return [];
  const startKey = dayKeyInZone(periodStart, timezone);
  const startMs = new Date(`${startKey}T00:00:00.000Z`).getTime();
  const buckets = new Map<number, number[]>();
  for (const p of points) {
    const dayKey = dayKeyInZone(p.takenAt, timezone);
    const dayMs = new Date(`${dayKey}T00:00:00.000Z`).getTime();
    const dayIndex = Math.floor((dayMs - startMs) / 86_400_000);
    const weekIndex = Math.floor(dayIndex / 7);
    const arr = buckets.get(weekIndex) ?? [];
    arr.push(p.valueMgdl);
    buckets.set(weekIndex, arr);
  }
  const sortedIndices = [...buckets.keys()].sort((a, b) => a - b);
  return sortedIndices.map((weekIndex) => {
    const values = buckets.get(weekIndex) ?? [];
    const weekStartMs = startMs + weekIndex * 7 * 86_400_000;
    return {
      weekStart: new Date(weekStartMs).toISOString().slice(0, 10),
      count: values.length,
      averageMgdl: mean(values),
      sdMgdl: stdDev(values),
    };
  });
}

export function summarizeGlucose(
  points: readonly GlucosePoint[],
  range: TargetRange,
  timezone: string,
  periodStart: Date,
  periodEnd: Date,
): GlucoseSummary {
  const values = valuesOf(points);
  const gaps = gapsMinutes(points);
  const dayKeys = new Set(points.map((p) => dayKeyInZone(p.takenAt, timezone)));
  const spanDays = Math.max(1, (periodEnd.getTime() - periodStart.getTime()) / 86_400_000);

  return {
    count: points.length,
    averageMgdl: mean(values),
    medianMgdl: median(values),
    sdMgdl: stdDev(values),
    cv: coefficientOfVariation(values),
    percentInRange: percentInBand(points, range, 'in'),
    percentBelowRange: percentInBand(points, range, 'below'),
    percentAboveRange: percentInBand(points, range, 'above'),
    readingsPerDay: points.length > 0 ? points.length / spanDays : null,
    daysWithReadings: dayKeys.size,
    averageGapMinutes: mean(gaps),
    medianGapMinutes: median(gaps),
    maxGapMinutes: gaps.length > 0 ? Math.max(...gaps) : null,
    histogram: histogram(points),
    byHourOfDay: byHourOfDay(points, timezone),
    byWeekday: byWeekday(points, timezone),
    byContext: byContext(points, range),
    morningVsEvening: compareMorningVsEvening(points, timezone),
    weekOverWeek: weekOverWeek(points, timezone, periodStart),
  };
}

export { quantile };
