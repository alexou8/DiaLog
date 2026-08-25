/**
 * Robust anomaly detection against the user's OWN baseline.
 *
 * Rather than compare a reading to a population reference range (which is
 * what the safety bands in lib/domain/thresholds.ts already do), this
 * flags readings that are unusual *for this person, at this time of day*
 * — e.g. a fasting reading well outside their own typical fasting range.
 *
 * Uses the median + MAD-based "modified z-score" (Iglewicz & Hoaglin,
 * 1993), which is robust to the outliers it is trying to detect (unlike
 * mean/SD-based z-scores, where a handful of extreme values inflate the SD
 * and mask themselves). |modified z| > 3.5 is the threshold recommended in
 * that paper and is what we use as the default.
 */
import { hourInZone, timeOfDayBucket } from '@/lib/domain/time';
import { modifiedZScores } from '../stats';
import type { GlucosePoint } from '../types';

export type TimeOfDayBucket = ReturnType<typeof timeOfDayBucket>;

export interface AnomalyFlag {
  readingId: string;
  takenAt: Date;
  valueMgdl: number;
  bucket: TimeOfDayBucket;
  modifiedZScore: number;
  /** Median of the baseline this reading was scored against. */
  baselineMedianMgdl: number;
  /** Number of readings (including this one) in the baseline bucket. */
  baselineSize: number;
}

export interface AnomalyOptions {
  timezone: string;
  /** Minimum number of same-bucket readings required before scoring is attempted. Default 10. */
  minBaselineSize?: number;
  /** |modified z-score| threshold to flag. Default 3.5 (Iglewicz & Hoaglin). */
  zThreshold?: number;
}

export function detectAnomalies(glucose: readonly GlucosePoint[], options: AnomalyOptions): AnomalyFlag[] {
  const minBaselineSize = options.minBaselineSize ?? 10;
  const zThreshold = options.zThreshold ?? 3.5;

  const byBucket = new Map<TimeOfDayBucket, GlucosePoint[]>();
  for (const g of glucose) {
    const bucket = timeOfDayBucket(hourInZone(g.takenAt, options.timezone));
    const arr = byBucket.get(bucket) ?? [];
    arr.push(g);
    byBucket.set(bucket, arr);
  }

  const flags: AnomalyFlag[] = [];
  for (const [bucket, points] of byBucket) {
    if (points.length < minBaselineSize) continue; // not enough same-bucket history to judge "unusual"
    const values = points.map((p) => p.valueMgdl);
    const scores = modifiedZScores(values, values);
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    const baselineMedian =
      sorted.length % 2 === 0 ? ((sorted[mid - 1] ?? 0) + (sorted[mid] ?? 0)) / 2 : (sorted[mid] ?? 0);

    for (let i = 0; i < points.length; i++) {
      const score = scores[i];
      const point = points[i];
      if (score === undefined || score === null || point === undefined) continue;
      if (Math.abs(score) > zThreshold) {
        flags.push({
          readingId: point.id,
          takenAt: point.takenAt,
          valueMgdl: point.valueMgdl,
          bucket,
          modifiedZScore: score,
          baselineMedianMgdl: baselineMedian,
          baselineSize: points.length,
        });
      }
    }
  }

  flags.sort((a, b) => Math.abs(b.modifiedZScore) - Math.abs(a.modifiedZScore));
  return flags;
}
