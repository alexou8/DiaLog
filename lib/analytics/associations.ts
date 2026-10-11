/**
 * Behaviour <-> glucose association analyses.
 *
 * Every function here returns a `Finding` (see lib/domain/evidence.ts) or
 * `null` when there isn't enough data. Statements are strictly
 * observational — "your average X was Y on days you did Z" — never
 * causal ("Z caused Y") and never prescriptive ("you should do Z").
 *
 * A recurring design choice: where an analysis compares groups of *days*
 * (activity, sleep, stress), we aggregate to one value per day before
 * running the test, rather than treating every raw reading as an
 * independent sample. Multiple readings from the same day are correlated
 * with each other (pseudoreplication), and per-day aggregation avoids
 * quietly inflating the apparent sample size / significance.
 */
import { dayKeyInZone, weekdayInZone } from '@/lib/domain/time';
import { POST_MEAL_WINDOW_MIN } from '@/lib/domain/thresholds';
import {
  gradeComparisonEvidence,
  gradeEvidence,
  type EvidenceLevel,
  type Finding,
} from '@/lib/domain/evidence';
import type { AnalyticsInput, GlucosePoint } from './types';
import { formatDelta, formatLevel } from './format';
import { cohensD, mean, welchTTest, type WelchTTestResult } from './stats';

function isoDay(d: Date): string {
  return d.toISOString();
}

function periodBounds(input: AnalyticsInput): { periodStart: string; periodEnd: string } {
  return { periodStart: isoDay(input.periodStart), periodEnd: isoDay(input.periodEnd) };
}

interface GlucoseTimeIndex {
  /** Epoch ms, ascending. */
  times: Float64Array;
  values: Float64Array;
}

/**
 * Glucose sorted by time once per input array, so a window lookup is a
 * binary search instead of a scan. The old per-meal `glucose.filter(...)`
 * made the post-meal and sleep analyses O(events × readings): ~9.4M
 * comparisons plus an array allocation per event over 90 days of CGM.
 * Cached by array identity because `runAssociations` asks several analyses
 * about the same input.
 */
const glucoseIndexCache = new WeakMap<readonly GlucosePoint[], GlucoseTimeIndex>();

function glucoseIndex(glucose: readonly GlucosePoint[]): GlucoseTimeIndex {
  const cached = glucoseIndexCache.get(glucose);
  if (cached) return cached;
  let sorted = true;
  for (let i = 1; i < glucose.length; i++) {
    if (glucose[i]!.takenAt.getTime() < glucose[i - 1]!.takenAt.getTime()) {
      sorted = false;
      break;
    }
  }
  // The database returns readings ascending, but other callers may not.
  const ordered = sorted
    ? glucose
    : [...glucose].sort((a, b) => a.takenAt.getTime() - b.takenAt.getTime());
  const index: GlucoseTimeIndex = {
    times: new Float64Array(ordered.length),
    values: new Float64Array(ordered.length),
  };
  ordered.forEach((g, i) => {
    index.times[i] = g.takenAt.getTime();
    index.values[i] = g.valueMgdl;
  });
  glucoseIndexCache.set(glucose, index);
  return index;
}

/** First index whose time is >= `t`. */
function lowerBound(times: Float64Array, t: number): number {
  let lo = 0;
  let hi = times.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (times[mid]! < t) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/** Mean of readings with startMs <= takenAt <= endMs, or null when there are none. */
function windowMean(
  glucose: readonly GlucosePoint[],
  startMs: number,
  endMs: number,
): number | null {
  const { times, values } = glucoseIndex(glucose);
  let total = 0;
  let count = 0;
  for (let i = lowerBound(times, startMs); i < times.length && times[i]! <= endMs; i++) {
    const v = values[i]!;
    if (!Number.isFinite(v)) continue;
    total += v;
    count++;
  }
  return count === 0 ? null : total / count;
}

/** Average of readings in the post-meal response window (60-180 min, per lib/domain/thresholds) after `anchor`. */
function postWindowAverage(glucose: readonly GlucosePoint[], anchor: Date): number | null {
  return windowMean(
    glucose,
    anchor.getTime() + POST_MEAL_WINDOW_MIN.start * 60_000,
    anchor.getTime() + POST_MEAL_WINDOW_MIN.end * 60_000,
  );
}

/**
 * Test, effect size and evidence grade for one two-group comparison. The
 * grade accounts for the smaller group and for whether the difference is
 * distinguishable from noise (see `gradeComparisonEvidence`).
 */
function compareGroups(
  a: readonly number[],
  b: readonly number[],
  analysis: string,
): { test: WelchTTestResult | null; d: number | null; evidenceLevel: EvidenceLevel } {
  const test = welchTTest(a, b);
  const d = cohensD(a, b);
  const evidenceLevel = gradeComparisonEvidence(
    { nA: a.length, nB: b.length, pValue: test?.pValue ?? null, effectSize: d },
    analysis,
  );
  return { test, d, evidenceLevel };
}

/** The metrics every comparison finding reports about its test. */
function testMetrics(test: WelchTTestResult | null, d: number | null) {
  return {
    welchT: test?.t ?? null,
    welchPValue: test?.pValue ?? null,
    differenceCiLowMgdl: test?.ciLow ?? null,
    differenceCiHighMgdl: test?.ciHigh ?? null,
    cohensD: d,
  };
}

export type CarbBucket = 'low' | 'medium' | 'high';

export function carbBucketOf(carbsG: number): CarbBucket {
  if (carbsG < 30) return 'low';
  if (carbsG < 60) return 'medium';
  return 'high';
}

/**
 * Post-meal glucose response grouped by how many carbs the meal contained.
 * One sample per meal (the mean of that meal's readings in the 60-180 min
 * window), not one sample per reading.
 */
export function postMealResponseByCarbBucket(input: AnalyticsInput): Finding | null {
  const buckets: Record<CarbBucket, number[]> = { low: [], medium: [], high: [] };
  for (const meal of input.meals) {
    if (meal.carbsG === null || !Number.isFinite(meal.carbsG)) continue;
    const avg = postWindowAverage(input.glucose, meal.takenAt);
    if (avg === null) continue;
    buckets[carbBucketOf(meal.carbsG)].push(avg);
  }
  const totalN = buckets.low.length + buckets.medium.length + buckets.high.length;
  if (gradeEvidence(totalN, 'association') === 'INSUFFICIENT') return null;
  if (buckets.low.length < 2 && buckets.high.length < 2) return null;

  const lowAvg = mean(buckets.low);
  const medAvg = mean(buckets.medium);
  const highAvg = mean(buckets.high);
  // The statement describes every bucket, but the only comparison tested is
  // high vs low, so the grade follows that comparison. Without both arms it
  // is a description, not a pattern, so it stays an early signal.
  const comparable = buckets.low.length >= 2 && buckets.high.length >= 2;
  const compared = comparable
    ? compareGroups(buckets.high, buckets.low, 'association')
    : { test: null, d: null, evidenceLevel: 'EARLY' as EvidenceLevel };
  const { test, d } = compared;
  // The descriptive gate above already passed, so the finding exists; the
  // comparison only decides how strongly it is graded. A comparison graded on
  // its smaller arm (2 × min) can be INSUFFICIENT where the description is
  // not — 20 low vs 3 high — and returning null there made the finding vanish
  // when two more high-carb meals turned 20 vs 1 (described, EARLY) into
  // 20 vs 3 (compared, INSUFFICIENT). More data must never remove a finding,
  // so a too-small comparison stays an early signal, like the incomparable
  // case.
  const evidenceLevel: EvidenceLevel =
    compared.evidenceLevel === 'INSUFFICIENT' ? 'EARLY' : compared.evidenceLevel;

  const parts: string[] = [];
  if (lowAvg !== null)
    parts.push(
      `lower-carb meals (<30g, n=${buckets.low.length}) averaged ${formatLevel(lowAvg, input.displayUnit)} 1-3h after eating`,
    );
  if (medAvg !== null)
    parts.push(
      `medium-carb meals (30-60g, n=${buckets.medium.length}) averaged ${formatLevel(medAvg, input.displayUnit)}`,
    );
  if (highAvg !== null)
    parts.push(
      `higher-carb meals (60g+, n=${buckets.high.length}) averaged ${formatLevel(highAvg, input.displayUnit)}`,
    );

  return {
    id: `post-meal-carb-bucket:${periodBounds(input).periodStart}:${periodBounds(input).periodEnd}`,
    kind: 'post-meal-carb-bucket',
    statement: `In your logs, ${parts.join('; ')}.`,
    sampleSize: totalN,
    evidenceLevel,
    source: 'STATISTICAL',
    metrics: {
      lowAvgMgdl: lowAvg,
      mediumAvgMgdl: medAvg,
      highAvgMgdl: highAvg,
      lowN: buckets.low.length,
      mediumN: buckets.medium.length,
      highN: buckets.high.length,
      ...testMetrics(test, d),
    },
    basis: `Meals with a logged carb amount, compared against your glucose readings ${POST_MEAL_WINDOW_MIN.start}-${POST_MEAL_WINDOW_MIN.end} minutes afterward, one average per meal.`,
    ...periodBounds(input),
    caveats: [
      'Carb estimates you log may not exactly match what was eaten.',
      'Other things eaten or done around the same meal were not accounted for.',
    ],
  };
}

/** Local calendar-day key that a Date falls on. */
function dayKey(d: Date, tz: string): string {
  return dayKeyInZone(d, tz);
}

/**
 * Post-dinner glucose on days you logged activity after dinner vs days you
 * did not. Activity = an exercise session starting within 4 hours after
 * the day's (last) dinner meal.
 */
export function postDinnerActivityComparison(input: AnalyticsInput): Finding | null {
  const dinners = new Map<string, Date>(); // dayKey -> last dinner time that day
  for (const meal of input.meals) {
    if (meal.mealType !== 'DINNER') continue;
    const key = dayKey(meal.takenAt, input.timezone);
    const existing = dinners.get(key);
    if (!existing || meal.takenAt.getTime() > existing.getTime()) dinners.set(key, meal.takenAt);
  }

  const withActivity: number[] = [];
  const withoutActivity: number[] = [];
  for (const [, dinnerTime] of dinners) {
    const avg = postWindowAverage(input.glucose, dinnerTime);
    if (avg === null) continue;
    const hadActivity = input.exercise.some((e) => {
      const t = e.takenAt.getTime();
      return t >= dinnerTime.getTime() && t <= dinnerTime.getTime() + 4 * 60 * 60_000;
    });
    (hadActivity ? withActivity : withoutActivity).push(avg);
  }

  const totalN = withActivity.length + withoutActivity.length;
  const { test, d, evidenceLevel } = compareGroups(withoutActivity, withActivity, 'comparison');
  if (evidenceLevel === 'INSUFFICIENT') return null;
  const avgWith = mean(withActivity);
  const avgWithout = mean(withoutActivity);

  const diffText =
    avgWith !== null && avgWithout !== null
      ? `${formatDelta(avgWithout - avgWith, input.displayUnit)} ${avgWithout > avgWith ? 'lower' : 'higher'} on days you logged activity after dinner`
      : 'different';

  return {
    id: `post-dinner-activity:${periodBounds(input).periodStart}:${periodBounds(input).periodEnd}`,
    kind: 'post-dinner-activity',
    statement: `Your average post-dinner reading was ${diffText} (n=${withActivity.length} active days vs n=${withoutActivity.length} other days).`,
    sampleSize: totalN,
    evidenceLevel,
    source: 'STATISTICAL',
    metrics: {
      avgWithActivityMgdl: avgWith,
      avgWithoutActivityMgdl: avgWithout,
      nWithActivity: withActivity.length,
      nWithoutActivity: withoutActivity.length,
      ...testMetrics(test, d),
    },
    basis:
      'Days with a logged dinner, comparing your average post-dinner glucose (60-180 min after) on days with vs without a logged activity in the 4 hours after.',
    ...periodBounds(input),
    caveats: [
      'These days may differ in other ways that were not recorded (what was eaten, sleep, stress).',
      'Only logged activity is counted. Unlogged activity is treated as none.',
    ],
  };
}

/** Readings after nights of short (<6h) vs adequate (>=7h) sleep, using the morning glucose average that follows. */
export function sleepDurationComparison(input: AnalyticsInput): Finding | null {
  const shortDayAverages: number[] = [];
  const adequateDayAverages: number[] = [];

  for (const s of input.sleep) {
    const hours = s.durationMin / 60;
    if (hours >= 6 && hours < 7) continue; // ambiguous middle band, excluded for a cleaner comparison
    const morningStart = s.endedAt.getTime();
    const morningEnd = morningStart + 6 * 60 * 60_000; // 6h window after waking
    const avg = windowMean(input.glucose, morningStart, morningEnd);
    if (avg === null) continue;
    (hours < 6 ? shortDayAverages : adequateDayAverages).push(avg);
  }

  const totalN = shortDayAverages.length + adequateDayAverages.length;
  const { test, d, evidenceLevel } = compareGroups(
    shortDayAverages,
    adequateDayAverages,
    'comparison',
  );
  if (evidenceLevel === 'INSUFFICIENT') return null;
  const avgShort = mean(shortDayAverages);
  const avgAdequate = mean(adequateDayAverages);

  return {
    id: `sleep-duration:${periodBounds(input).periodStart}:${periodBounds(input).periodEnd}`,
    kind: 'sleep-duration',
    statement: `Your average glucose in the morning after under 6 hours of sleep was ${avgShort !== null ? formatLevel(avgShort, input.displayUnit) : 'an unknown amount'} (n=${shortDayAverages.length}), vs ${avgAdequate !== null ? formatLevel(avgAdequate, input.displayUnit) : 'an unknown amount'} after 7+ hours (n=${adequateDayAverages.length}).`,
    sampleSize: totalN,
    evidenceLevel,
    source: 'STATISTICAL',
    metrics: {
      avgAfterShortSleepMgdl: avgShort,
      avgAfterAdequateSleepMgdl: avgAdequate,
      nShort: shortDayAverages.length,
      nAdequate: adequateDayAverages.length,
      ...testMetrics(test, d),
    },
    basis:
      'Nights with a logged sleep session, comparing your average glucose in the 6 hours after waking on nights under 6h sleep vs nights of 7h or more.',
    ...periodBounds(input),
    caveats: [
      'Sleep duration is self-logged and may not reflect actual sleep quality.',
      'These mornings may differ in other ways that were not recorded.',
    ],
  };
}

/** Fasting-context readings on weekdays vs weekends (local time). */
export function fastingWeekdayVsWeekend(input: AnalyticsInput): Finding | null {
  const weekday: number[] = [];
  const weekend: number[] = [];
  for (const g of input.glucose) {
    if (g.context !== 'FASTING') continue;
    const wd = weekdayInZone(g.takenAt, input.timezone);
    (wd === 0 || wd === 6 ? weekend : weekday).push(g.valueMgdl);
  }
  const totalN = weekday.length + weekend.length;
  const { test, d, evidenceLevel } = compareGroups(weekend, weekday, 'comparison');
  if (evidenceLevel === 'INSUFFICIENT') return null;
  const avgWeekday = mean(weekday);
  const avgWeekend = mean(weekend);

  return {
    id: `fasting-weekday-weekend:${periodBounds(input).periodStart}:${periodBounds(input).periodEnd}`,
    kind: 'fasting-weekday-weekend',
    statement: `Your average fasting reading was ${avgWeekday !== null ? formatLevel(avgWeekday, input.displayUnit) : 'an unknown amount'} on weekdays (n=${weekday.length}) vs ${avgWeekend !== null ? formatLevel(avgWeekend, input.displayUnit) : 'an unknown amount'} on weekends (n=${weekend.length}).`,
    sampleSize: totalN,
    evidenceLevel,
    source: 'STATISTICAL',
    metrics: {
      avgWeekdayMgdl: avgWeekday,
      avgWeekendMgdl: avgWeekend,
      nWeekday: weekday.length,
      nWeekend: weekend.length,
      ...testMetrics(test, d),
    },
    basis:
      'Readings you logged with a "Fasting" context, split by whether the local day was a weekday or weekend day.',
    ...periodBounds(input),
    caveats: [
      'Weekday and weekend routines (meals, sleep timing, activity) commonly differ in ways that were not recorded.',
    ],
  };
}

/** Days classified as high-stress vs low-stress (from logged mood entries) compared on average glucose that day. */
export function stressComparison(input: AnalyticsInput): Finding | null {
  const stressByDay = new Map<string, number[]>();
  for (const m of input.moods) {
    if (m.stress === null) continue;
    const key = dayKey(m.takenAt, input.timezone);
    const arr = stressByDay.get(key) ?? [];
    arr.push(m.stress);
    stressByDay.set(key, arr);
  }

  const glucoseByDay = new Map<string, number[]>();
  for (const g of input.glucose) {
    const key = dayKey(g.takenAt, input.timezone);
    const arr = glucoseByDay.get(key) ?? [];
    arr.push(g.valueMgdl);
    glucoseByDay.set(key, arr);
  }

  const highStressDayAverages: number[] = [];
  const lowStressDayAverages: number[] = [];
  for (const [key, stressValues] of stressByDay) {
    const avgStress = mean(stressValues);
    if (avgStress === null) continue;
    const glucoseValues = glucoseByDay.get(key);
    const avgGlucose = glucoseValues ? mean(glucoseValues) : null;
    if (avgGlucose === null) continue;
    if (avgStress >= 4) highStressDayAverages.push(avgGlucose);
    else if (avgStress <= 2) lowStressDayAverages.push(avgGlucose);
  }

  const totalN = highStressDayAverages.length + lowStressDayAverages.length;
  const { test, d, evidenceLevel } = compareGroups(
    highStressDayAverages,
    lowStressDayAverages,
    'comparison',
  );
  if (evidenceLevel === 'INSUFFICIENT') return null;
  const avgHigh = mean(highStressDayAverages);
  const avgLow = mean(lowStressDayAverages);

  return {
    id: `stress:${periodBounds(input).periodStart}:${periodBounds(input).periodEnd}`,
    kind: 'stress',
    statement: `Your average glucose was ${avgHigh !== null ? formatLevel(avgHigh, input.displayUnit) : 'an unknown amount'} on days you logged higher stress (n=${highStressDayAverages.length}), vs ${avgLow !== null ? formatLevel(avgLow, input.displayUnit) : 'an unknown amount'} on lower-stress days (n=${lowStressDayAverages.length}).`,
    sampleSize: totalN,
    evidenceLevel,
    source: 'STATISTICAL',
    metrics: {
      avgHighStressMgdl: avgHigh,
      avgLowStressMgdl: avgLow,
      nHighStress: highStressDayAverages.length,
      nLowStress: lowStressDayAverages.length,
      ...testMetrics(test, d),
    },
    basis:
      'Days with a logged mood/stress entry (stress 1-5), comparing your average glucose that day on high-stress days (4-5) vs low-stress days (1-2).',
    ...periodBounds(input),
    caveats: [
      'Stress is self-rated and days may differ in other unrecorded ways (sleep, meals, activity).',
      'Only one stress rating scale (1-5) is used; days with mid-range stress (3) are excluded from this comparison.',
    ],
  };
}

/** Runs every association analysis and returns only the non-null Findings. */
export function runAssociations(input: AnalyticsInput): Finding[] {
  const findings = [
    postMealResponseByCarbBucket(input),
    postDinnerActivityComparison(input),
    sleepDurationComparison(input),
    fastingWeekdayVsWeekend(input),
    stressComparison(input),
  ];
  return findings.filter((f): f is Finding => f !== null);
}
