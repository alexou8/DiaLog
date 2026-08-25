/**
 * Per-reading feature engineering, ported from (and improved on) the legacy
 * `ml/src/features.py`. The legacy version used `pandas.searchsorted` for a
 * last-event-before lookup and then silently imputed missing values with
 * the column median — that hides "we don't actually know" behind a
 * plausible-looking number. This port keeps the merge-join idea (each
 * event stream is scanned once, in step with the sorted glucose readings,
 * for O(n log n) total including the sort) but leaves genuinely missing
 * values as `null` for downstream code to handle explicitly.
 */
import { hourInZone, minutesBetween, timeOfDayBucket, weekdayInZone } from '@/lib/domain/time';
import type { ExercisePoint, GlucosePoint, MealPoint, MedicationPoint, SleepPoint } from './types';

/** Mirrors the return type of `timeOfDayBucket` in lib/domain/time.ts. */
export type TimeOfDayBucket = ReturnType<typeof timeOfDayBucket>;

export interface GlucoseFeatureRow {
  glucoseId: string;
  takenAt: Date;
  valueMgdl: number;
  minutesSinceMeal: number | null;
  lastMealCarbsG: number | null;
  minutesSinceMedication: number | null;
  minutesSinceExercise: number | null;
  /** Minutes of logged exercise whose start fell in the 3 hours before this reading. */
  exerciseMinutesPrior3h: number;
  hour: number;
  weekday: number;
  timeOfDay: TimeOfDayBucket;
  sleepHoursPriorNight: number | null;
}

export interface FeatureOptions {
  timezone: string;
  /** How far back a meal can be and still count as "the last meal". Default 6h, matches the legacy lookback. */
  mealLookbackMin?: number;
  /** How far back a medication dose can be and still count. Default 6h. */
  medicationLookbackMin?: number;
  /** How far back a completed exercise session can be and still count for "minutes since exercise". Default 24h. */
  exerciseLookbackMin?: number;
  /** How stale a completed sleep session can be and still count as "prior night". Default 18h. */
  sleepLookbackMin?: number;
}

const DEFAULTS = {
  mealLookbackMin: 360,
  medicationLookbackMin: 360,
  exerciseLookbackMin: 24 * 60,
  sleepLookbackMin: 18 * 60,
} as const;

function exerciseEnd(e: ExercisePoint): Date {
  return e.endedAt ?? new Date(e.takenAt.getTime() + e.durationMin * 60_000);
}

/**
 * Computes engineered features for every glucose reading. Each event
 * stream (meals, medications, exercise, sleep) is sorted once and then
 * walked forward with a single pointer in lockstep with the (also sorted)
 * glucose readings, so the whole function is O(n log n) rather than the
 * O(n * m) nested-loop scan a naive port of the Python would produce.
 *
 * The returned array is in the same order as the input `glucose` array.
 */
export function computeGlucoseFeatures(
  glucose: readonly GlucosePoint[],
  meals: readonly MealPoint[],
  medications: readonly MedicationPoint[],
  exercise: readonly ExercisePoint[],
  sleep: readonly SleepPoint[],
  options: FeatureOptions,
): GlucoseFeatureRow[] {
  const opts = { ...DEFAULTS, ...options };

  // Sort a copy of glucose with original-index tracking so we can return
  // results in the caller's original order.
  const orderedGlucose = glucose.map((g, originalIndex) => ({ g, originalIndex }));
  orderedGlucose.sort((a, b) => a.g.takenAt.getTime() - b.g.takenAt.getTime());

  const sortedMeals = [...meals].sort((a, b) => a.takenAt.getTime() - b.takenAt.getTime());
  const sortedMeds = [...medications].sort((a, b) => a.takenAt.getTime() - b.takenAt.getTime());
  const sortedExercise = [...exercise].sort((a, b) => a.takenAt.getTime() - b.takenAt.getTime());
  const sortedSleep = [...sleep].sort(
    (a, b) => exerciseSleepEnd(a).getTime() - exerciseSleepEnd(b).getTime(),
  );

  let mealPtr = -1; // index of last meal with takenAt <= ts
  let medPtr = -1;
  let exLastEndedPtr = -1; // index of last exercise session whose end <= ts (assumes non-overlapping, chronological sessions)
  let exWindowStart = 0; // sliding window [exWindowStart, exWindowEnd) of sessions with takenAt in (ts-3h, ts]
  let exWindowEnd = 0;
  let exWindowSum = 0;
  let sleepPtr = -1; // index of last sleep session with endedAt <= ts

  const results = new Array<GlucoseFeatureRow>(glucose.length);

  for (const { g, originalIndex } of orderedGlucose) {
    const ts = g.takenAt.getTime();

    while (mealPtr + 1 < sortedMeals.length && sortedMeals[mealPtr + 1]!.takenAt.getTime() <= ts)
      mealPtr++;
    const lastMeal = mealPtr >= 0 ? sortedMeals[mealPtr] : undefined;
    let minutesSinceMeal: number | null = null;
    let lastMealCarbsG: number | null = null;
    if (lastMeal) {
      const mins = minutesBetween(lastMeal.takenAt, g.takenAt);
      if (mins <= opts.mealLookbackMin) {
        minutesSinceMeal = mins;
        lastMealCarbsG = lastMeal.carbsG;
      }
    }

    while (medPtr + 1 < sortedMeds.length && sortedMeds[medPtr + 1]!.takenAt.getTime() <= ts)
      medPtr++;
    const lastMed = medPtr >= 0 ? sortedMeds[medPtr] : undefined;
    let minutesSinceMedication: number | null = null;
    if (lastMed) {
      const mins = minutesBetween(lastMed.takenAt, g.takenAt);
      if (mins <= opts.medicationLookbackMin) minutesSinceMedication = mins;
    }

    // "Last exercise ended before ts" — assumes sessions are chronological
    // and non-overlapping, which holds for logged real-world exercise.
    while (
      exLastEndedPtr + 1 < sortedExercise.length &&
      exerciseEnd(sortedExercise[exLastEndedPtr + 1]!).getTime() <= ts
    ) {
      exLastEndedPtr++;
    }
    const lastExercise = exLastEndedPtr >= 0 ? sortedExercise[exLastEndedPtr] : undefined;
    let minutesSinceExercise: number | null = null;
    if (lastExercise) {
      const mins = minutesBetween(exerciseEnd(lastExercise), g.takenAt);
      if (mins <= opts.exerciseLookbackMin) minutesSinceExercise = mins;
    }

    // Sliding 3h window of exercise minutes, keyed on session start time.
    while (
      exWindowEnd < sortedExercise.length &&
      sortedExercise[exWindowEnd]!.takenAt.getTime() <= ts
    ) {
      exWindowSum += sortedExercise[exWindowEnd]!.durationMin;
      exWindowEnd++;
    }
    while (
      exWindowStart < exWindowEnd &&
      sortedExercise[exWindowStart]!.takenAt.getTime() < ts - 180 * 60_000
    ) {
      exWindowSum -= sortedExercise[exWindowStart]!.durationMin;
      exWindowStart++;
    }

    while (sleepPtr + 1 < sortedSleep.length && sortedSleep[sleepPtr + 1]!.endedAt.getTime() <= ts)
      sleepPtr++;
    const lastSleep = sleepPtr >= 0 ? sortedSleep[sleepPtr] : undefined;
    let sleepHoursPriorNight: number | null = null;
    if (lastSleep) {
      const mins = minutesBetween(lastSleep.endedAt, g.takenAt);
      if (mins <= opts.sleepLookbackMin) sleepHoursPriorNight = lastSleep.durationMin / 60;
    }

    const hour = hourInZone(g.takenAt, opts.timezone);

    results[originalIndex] = {
      glucoseId: g.id,
      takenAt: g.takenAt,
      valueMgdl: g.valueMgdl,
      minutesSinceMeal,
      lastMealCarbsG,
      minutesSinceMedication,
      minutesSinceExercise,
      exerciseMinutesPrior3h: exWindowSum,
      hour,
      weekday: weekdayInZone(g.takenAt, opts.timezone),
      timeOfDay: timeOfDayBucket(hour),
      sleepHoursPriorNight,
    };
  }

  return results;
}

function exerciseSleepEnd(s: SleepPoint): Date {
  return s.endedAt;
}
