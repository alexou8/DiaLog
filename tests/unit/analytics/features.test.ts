import { describe, expect, it } from 'vitest';
import { computeGlucoseFeatures } from '@/lib/analytics/features';
import type { ExercisePoint, GlucosePoint, MealPoint, MedicationPoint, SleepPoint } from '@/lib/analytics/types';

const TZ = 'UTC';

function glucose(id: string, iso: string, valueMgdl = 120): GlucosePoint {
  return { id, takenAt: new Date(iso), valueMgdl, context: 'RANDOM' };
}
function meal(id: string, iso: string, carbsG: number | null): MealPoint {
  return { id, takenAt: new Date(iso), mealType: 'LUNCH', carbsG, description: 'test meal' };
}
function med(id: string, iso: string): MedicationPoint {
  return { id, takenAt: new Date(iso), name: 'metformin' };
}
function exercise(id: string, iso: string, durationMin: number, endedIso?: string): ExercisePoint {
  return { id, takenAt: new Date(iso), endedAt: endedIso ? new Date(endedIso) : null, durationMin, activity: 'walk', intensity: 'LIGHT' };
}
function sleep(id: string, startIso: string, endIso: string, durationMin: number): SleepPoint {
  return { id, takenAt: new Date(startIso), endedAt: new Date(endIso), durationMin, quality: null };
}

describe('computeGlucoseFeatures', () => {
  it('returns an empty array for no glucose readings', () => {
    expect(computeGlucoseFeatures([], [], [], [], [], { timezone: TZ })).toEqual([]);
  });

  it('finds minutes-since-meal and last-meal-carbs within the lookback window', () => {
    const g = [glucose('g1', '2026-01-01T13:00:00Z')]; // 60 min after meal
    const meals = [meal('m1', '2026-01-01T12:00:00Z', 45)];
    const [row] = computeGlucoseFeatures(g, meals, [], [], [], { timezone: TZ });
    expect(row?.minutesSinceMeal).toBeCloseTo(60);
    expect(row?.lastMealCarbsG).toBe(45);
  });

  it('leaves minutesSinceMeal null when the reading is outside the lookback window', () => {
    const g = [glucose('g1', '2026-01-01T20:00:00Z')]; // 8h after meal, lookback default 6h
    const meals = [meal('m1', '2026-01-01T12:00:00Z', 45)];
    const [row] = computeGlucoseFeatures(g, meals, [], [], [], { timezone: TZ });
    expect(row?.minutesSinceMeal).toBeNull();
    expect(row?.lastMealCarbsG).toBeNull();
  });

  it('never imputes a missing meal carb amount — null carbs stay null even when a matching meal is found', () => {
    const g = [glucose('g1', '2026-01-01T12:30:00Z')];
    const meals = [meal('m1', '2026-01-01T12:00:00Z', null)];
    const [row] = computeGlucoseFeatures(g, meals, [], [], [], { timezone: TZ });
    expect(row?.minutesSinceMeal).toBeCloseTo(30);
    expect(row?.lastMealCarbsG).toBeNull();
  });

  it('picks the most recent of several prior meals (merge-join correctness)', () => {
    const g = [glucose('g1', '2026-01-01T14:00:00Z')];
    const meals = [meal('m1', '2026-01-01T09:00:00Z', 20), meal('m2', '2026-01-01T13:30:00Z', 55), meal('m3', '2026-01-01T15:00:00Z', 99)];
    const [row] = computeGlucoseFeatures(g, meals, [], [], [], { timezone: TZ });
    expect(row?.lastMealCarbsG).toBe(55); // m3 is after the reading, must be ignored
  });

  it('computes minutes-since-medication independently of meals', () => {
    const g = [glucose('g1', '2026-01-01T13:00:00Z')];
    const meds = [med('d1', '2026-01-01T12:45:00Z')];
    const [row] = computeGlucoseFeatures(g, [], meds, [], [], { timezone: TZ });
    expect(row?.minutesSinceMedication).toBeCloseTo(15);
  });

  it('computes minutes-since-exercise from the session end time', () => {
    const g = [glucose('g1', '2026-01-01T13:00:00Z')];
    const ex = [exercise('e1', '2026-01-01T11:00:00Z', 30, '2026-01-01T11:30:00Z')];
    const [row] = computeGlucoseFeatures(g, [], [], ex, [], { timezone: TZ });
    expect(row?.minutesSinceExercise).toBeCloseTo(90);
  });

  it('falls back to takenAt + durationMin when endedAt is missing', () => {
    const g = [glucose('g1', '2026-01-01T12:00:00Z')];
    const ex = [exercise('e1', '2026-01-01T11:00:00Z', 30)]; // ends at 11:30 implicitly
    const [row] = computeGlucoseFeatures(g, [], [], ex, [], { timezone: TZ });
    expect(row?.minutesSinceExercise).toBeCloseTo(30);
  });

  it('sums exercise minutes in the prior 3 hours (sliding window)', () => {
    const g = [glucose('g1', '2026-01-01T15:00:00Z')];
    const ex = [
      exercise('e1', '2026-01-01T12:30:00Z', 20), // 2.5h before -> in window
      exercise('e2', '2026-01-01T14:00:00Z', 10), // 1h before -> in window
      exercise('e3', '2026-01-01T11:00:00Z', 40), // 4h before -> out of window
    ];
    const [row] = computeGlucoseFeatures(g, [], [], ex, [], { timezone: TZ });
    expect(row?.exerciseMinutesPrior3h).toBe(30);
  });

  it('computes sleepHoursPriorNight from the most recent completed sleep session', () => {
    const g = [glucose('g1', '2026-01-01T09:00:00Z')]; // 2h after waking
    const s = [sleep('s1', '2025-12-31T23:00:00Z', '2026-01-01T07:00:00Z', 480)];
    const [row] = computeGlucoseFeatures(g, [], [], [], s, { timezone: TZ });
    expect(row?.sleepHoursPriorNight).toBeCloseTo(8);
  });

  it('leaves sleepHoursPriorNight null when the last sleep session is too stale', () => {
    const g = [glucose('g1', '2026-01-02T09:00:00Z')]; // over 24h after that sleep ended
    const s = [sleep('s1', '2025-12-31T23:00:00Z', '2026-01-01T07:00:00Z', 480)];
    const [row] = computeGlucoseFeatures(g, [], [], [], s, { timezone: TZ });
    expect(row?.sleepHoursPriorNight).toBeNull();
  });

  it('preserves the input order of the glucose array in its output', () => {
    const g = [glucose('g2', '2026-01-01T14:00:00Z'), glucose('g1', '2026-01-01T12:00:00Z')];
    const rows = computeGlucoseFeatures(g, [], [], [], [], { timezone: TZ });
    expect(rows.map((r) => r.glucoseId)).toEqual(['g2', 'g1']);
  });

  it('computes hour, weekday, and timeOfDay from the given timezone', () => {
    const g = [glucose('g1', '2026-01-01T05:00:00Z')]; // Thursday 00:00 UTC
    const rows = computeGlucoseFeatures(g, [], [], [], [], { timezone: 'UTC' });
    expect(rows[0]?.hour).toBe(5);
    expect(rows[0]?.timeOfDay).toBe('overnight');
    expect(rows[0]?.weekday).toBe(4); // Thursday
  });
});
