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
import { gradeEvidence, type Finding } from '@/lib/domain/evidence';
import type { AnalyticsInput, GlucosePoint } from './types';
import { formatDelta, formatLevel } from './format';
import { cohensD, mean, welchTTest } from './stats';

function isoDay(d: Date): string {
  return d.toISOString();
}

function periodBounds(input: AnalyticsInput): { periodStart: string; periodEnd: string } {
  return { periodStart: isoDay(input.periodStart), periodEnd: isoDay(input.periodEnd) };
}

/** Average of readings in the post-meal response window (60-180 min, per lib/domain/thresholds) after `anchor`. */
function postWindowAverage(glucose: readonly GlucosePoint[], anchor: Date): number | null {
  const windowStartMs = anchor.getTime() + POST_MEAL_WINDOW_MIN.start * 60_000;
  const windowEndMs = anchor.getTime() + POST_MEAL_WINDOW_MIN.end * 60_000;
  const inWindow = glucose
    .filter((g) => g.takenAt.getTime() >= windowStartMs && g.takenAt.getTime() <= windowEndMs)
    .map((g) => g.valueMgdl);
  return mean(inWindow);
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
  const evidenceLevel = gradeEvidence(totalN, 'association');
  if (evidenceLevel === 'INSUFFICIENT') return null;
  if (buckets.low.length < 2 && buckets.high.length < 2) return null;

  const lowAvg = mean(buckets.low);
  const medAvg = mean(buckets.medium);
  const highAvg = mean(buckets.high);
  const test =
    buckets.low.length >= 2 && buckets.high.length >= 2
      ? welchTTest(buckets.high, buckets.low)
      : null;
  const d =
    buckets.low.length >= 2 && buckets.high.length >= 2 ? cohensD(buckets.high, buckets.low) : null;

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
      welchT: test?.t ?? null,
      welchPValue: test?.pValue ?? null,
      cohensD: d,
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
  const evidenceLevel = gradeEvidence(totalN, 'comparison');
  if (evidenceLevel === 'INSUFFICIENT' || withActivity.length < 2 || withoutActivity.length < 2)
    return null;

  const test = welchTTest(withoutActivity, withActivity);
  const d = cohensD(withoutActivity, withActivity);
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
      welchT: test?.t ?? null,
      welchPValue: test?.pValue ?? null,
      cohensD: d,
    },
    basis:
      'Days with a logged dinner, comparing your average post-dinner glucose (60-180 min after) on days with vs without a logged activity in the 4 hours after.',
    ...periodBounds(input),
    caveats: [
      'These days may differ in other ways that were not recorded (what was eaten, sleep, stress).',
      'Only logged activity is counted — unlogged activity is treated as none.',
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
    const morningValues = input.glucose
      .filter((g) => g.takenAt.getTime() >= morningStart && g.takenAt.getTime() <= morningEnd)
      .map((g) => g.valueMgdl);
    const avg = mean(morningValues);
    if (avg === null) continue;
    (hours < 6 ? shortDayAverages : adequateDayAverages).push(avg);
  }

  const totalN = shortDayAverages.length + adequateDayAverages.length;
  const evidenceLevel = gradeEvidence(totalN, 'comparison');
  if (
    evidenceLevel === 'INSUFFICIENT' ||
    shortDayAverages.length < 2 ||
    adequateDayAverages.length < 2
  )
    return null;

  const test = welchTTest(shortDayAverages, adequateDayAverages);
  const d = cohensD(shortDayAverages, adequateDayAverages);
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
      welchT: test?.t ?? null,
      welchPValue: test?.pValue ?? null,
      cohensD: d,
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
  const evidenceLevel = gradeEvidence(totalN, 'comparison');
  if (evidenceLevel === 'INSUFFICIENT' || weekday.length < 2 || weekend.length < 2) return null;

  const test = welchTTest(weekend, weekday);
  const d = cohensD(weekend, weekday);
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
      welchT: test?.t ?? null,
      welchPValue: test?.pValue ?? null,
      cohensD: d,
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
  const evidenceLevel = gradeEvidence(totalN, 'comparison');
  if (
    evidenceLevel === 'INSUFFICIENT' ||
    highStressDayAverages.length < 2 ||
    lowStressDayAverages.length < 2
  )
    return null;

  const test = welchTTest(highStressDayAverages, lowStressDayAverages);
  const d = cohensD(highStressDayAverages, lowStressDayAverages);
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
      welchT: test?.t ?? null,
      welchPValue: test?.pValue ?? null,
      cohensD: d,
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
