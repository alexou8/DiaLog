import { describe, expect, it } from 'vitest';
import { detectAnomalies } from '@/lib/analytics/ml/anomaly';
import {
  buildDayFeatureVectors,
  clusterDayPatterns,
  kmeans,
  mulberry32,
} from '@/lib/analytics/ml/cluster';
import { detectTrend } from '@/lib/analytics/ml/trend';
import { computeFeatureImportance } from '@/lib/analytics/ml/importance';
import type { AnalyticsInput, ExercisePoint, GlucosePoint, MealPoint } from '@/lib/analytics/types';
import { DEFAULT_TARGET_RANGE } from '@/lib/domain/thresholds';

const TZ = 'UTC';

function baseInput(overrides: Partial<AnalyticsInput> = {}): AnalyticsInput {
  return {
    glucose: [],
    meals: [],
    exercise: [],
    sleep: [],
    medications: [],
    moods: [],
    timezone: TZ,
    targetRange: DEFAULT_TARGET_RANGE,
    periodStart: new Date('2026-01-01T00:00:00Z'),
    periodEnd: new Date('2026-04-01T00:00:00Z'),
    ...overrides,
  };
}

function glucoseAt(iso: string, valueMgdl: number): GlucosePoint {
  return { id: iso + Math.random(), takenAt: new Date(iso), valueMgdl, context: 'RANDOM' };
}

describe('detectAnomalies', () => {
  it('returns no flags with too little baseline data', () => {
    const glucose = Array.from({ length: 5 }, (_, i) =>
      glucoseAt(`2026-01-0${i + 1}T12:00:00Z`, 100),
    );
    expect(detectAnomalies(glucose, { timezone: TZ })).toEqual([]);
  });

  it('flags a clear outlier within a well-populated time-of-day bucket', () => {
    const glucose: GlucosePoint[] = [];
    for (let day = 1; day <= 20; day++) {
      // Afternoon bucket (12:00-17:59 local == UTC here), tight around 110.
      glucose.push(
        glucoseAt(new Date(Date.UTC(2026, 0, day, 13, 0, 0)).toISOString(), 108 + (day % 3)),
      );
    }
    // One wildly different afternoon reading.
    glucose.push(glucoseAt(new Date(Date.UTC(2026, 0, 21, 13, 0, 0)).toISOString(), 320));

    const flags = detectAnomalies(glucose, { timezone: TZ, minBaselineSize: 10 });
    expect(flags.length).toBeGreaterThan(0);
    expect(flags[0]?.valueMgdl).toBe(320);
    expect(Math.abs(flags[0]!.modifiedZScore)).toBeGreaterThan(3.5);
  });

  it('handles empty input', () => {
    expect(detectAnomalies([], { timezone: TZ })).toEqual([]);
  });
});

describe('mulberry32', () => {
  it('is deterministic for a fixed seed', () => {
    const a = mulberry32(7);
    const b = mulberry32(7);
    const seqA = [a(), a(), a()];
    const seqB = [b(), b(), b()];
    expect(seqA).toEqual(seqB);
  });
});

describe('kmeans', () => {
  it('separates two well-clustered groups deterministically across repeated runs', () => {
    // Fixed points, not random ones, so the clustering assertion is genuinely
    // deterministic.
    const fixedPoints: number[][] = [
      [0, 0],
      [0.1, 0],
      [0, 0.1],
      [0.1, 0.1],
      [0.05, 0.05],
      [0, 0.05],
      [10, 10],
      [10.1, 10],
      [10, 10.1],
      [10.1, 10.1],
      [10.05, 10.05],
      [10, 10.05],
    ];
    const runA = kmeans(fixedPoints, 2, { seed: 42 });
    const runB = kmeans(fixedPoints, 2, { seed: 42 });
    expect(runA?.assignments).toEqual(runB?.assignments);
    // The two natural groups should end up in different clusters.
    const groupA = runA!.assignments.slice(0, 6);
    const groupB = runA!.assignments.slice(6);
    expect(new Set(groupA).size).toBe(1);
    expect(new Set(groupB).size).toBe(1);
    expect(groupA[0]).not.toBe(groupB[0]);
  });

  it('returns null for empty input', () => {
    expect(kmeans([], 2)).toBeNull();
  });
});

describe('buildDayFeatureVectors / clusterDayPatterns', () => {
  it('returns null when there are too few distinct days', () => {
    const glucose = [
      glucoseAt('2026-01-01T12:00:00Z', 100),
      glucoseAt('2026-01-02T12:00:00Z', 100),
    ];
    expect(clusterDayPatterns(baseInput({ glucose }), 3)).toBeNull();
  });

  it('builds one vector per local day with a glucose reading', () => {
    const glucose = [
      glucoseAt('2026-01-01T12:00:00Z', 100),
      glucoseAt('2026-01-01T18:00:00Z', 120),
      glucoseAt('2026-01-02T12:00:00Z', 90),
    ];
    const vectors = buildDayFeatureVectors(baseInput({ glucose }));
    expect(vectors).toHaveLength(2);
    expect(vectors[0]?.meanGlucoseMgdl).toBeCloseTo(110);
  });

  it('produces labelled clusters deterministically for a reproducible seed', () => {
    const glucose: GlucosePoint[] = [];
    const meals: MealPoint[] = [];
    const exercise: ExercisePoint[] = [];
    for (let day = 1; day <= 30; day++) {
      const highGlucoseDay = day % 2 === 0;
      const d = new Date(Date.UTC(2026, 0, day, 12, 0, 0));
      glucose.push(glucoseAt(d.toISOString(), highGlucoseDay ? 190 : 100));
      meals.push({
        id: `m${day}`,
        takenAt: d,
        mealType: 'LUNCH',
        carbsG: highGlucoseDay ? 90 : 20,
        description: '',
      });
      if (!highGlucoseDay) {
        exercise.push({
          id: `e${day}`,
          takenAt: d,
          endedAt: null,
          durationMin: 45,
          activity: 'run',
          intensity: 'MODERATE',
        });
      }
    }
    const input = baseInput({ glucose, meals, exercise });
    const clustersA = clusterDayPatterns(input, 2, 42);
    const clustersB = clusterDayPatterns(input, 2, 42);
    expect(clustersA).not.toBeNull();
    expect(clustersA).toEqual(clustersB); // deterministic given a fixed seed
    expect(clustersA!.every((c) => typeof c.label === 'string' && c.label.length > 0)).toBe(true);
  });
});

describe('detectTrend', () => {
  it('reports not-enough-data with too few days', () => {
    const glucose = [glucoseAt('2026-01-01T12:00:00Z', 100)];
    const result = detectTrend(baseInput({ glucose }));
    expect(result.classification).toBe('not-enough-data');
  });

  it('detects a clearly rising trend', () => {
    const glucose: GlucosePoint[] = [];
    for (let day = 1; day <= 25; day++) {
      glucose.push(
        glucoseAt(new Date(Date.UTC(2026, 0, day, 12, 0, 0)).toISOString(), 100 + day * 3),
      );
    }
    const result = detectTrend(baseInput({ glucose }));
    expect(result.classification).toBe('rising');
    expect(result.slopeMgdlPerDay).toBeGreaterThan(0);
  });

  it('detects a clearly improving (falling) trend', () => {
    const glucose: GlucosePoint[] = [];
    for (let day = 1; day <= 25; day++) {
      glucose.push(
        glucoseAt(new Date(Date.UTC(2026, 0, day, 12, 0, 0)).toISOString(), 200 - day * 3),
      );
    }
    const result = detectTrend(baseInput({ glucose }));
    expect(result.classification).toBe('improving');
    expect(result.slopeMgdlPerDay).toBeLessThan(0);
  });

  it('classifies a flat, noiseless series as stable', () => {
    const glucose: GlucosePoint[] = [];
    for (let day = 1; day <= 25; day++) {
      glucose.push(glucoseAt(new Date(Date.UTC(2026, 0, day, 12, 0, 0)).toISOString(), 110));
    }
    const result = detectTrend(baseInput({ glucose }));
    expect(result.classification).toBe('stable');
  });
});

describe('computeFeatureImportance', () => {
  it('refuses (returns null) below the model evidence threshold', () => {
    const glucose = [glucoseAt('2026-01-01T13:30:00Z', 150)];
    const meals: MealPoint[] = [
      {
        id: 'm1',
        takenAt: new Date('2026-01-01T12:00:00Z'),
        mealType: 'LUNCH',
        carbsG: 40,
        description: '',
      },
    ];
    expect(computeFeatureImportance(baseInput({ glucose, meals }))).toBeNull();
  });

  it('fits a personalised model once enough post-meal data is present, ranking carbs as important when it drives the outcome', () => {
    const glucose: GlucosePoint[] = [];
    const meals: MealPoint[] = [];
    for (let i = 0; i < 60; i++) {
      const mealTime = new Date(Date.UTC(2026, 0, 1 + Math.floor(i / 3), 8 + (i % 3) * 5, 0, 0));
      const carbs = 15 + (i % 6) * 15; // varies 15..90
      meals.push({
        id: `m${i}`,
        takenAt: mealTime,
        mealType: 'LUNCH',
        carbsG: carbs,
        description: '',
      });
      // Response is a clean linear function of carbs plus tiny noise, so carbs should dominate.
      const response = 100 + carbs * 1.2 + ((i % 5) - 2);
      glucose.push(glucoseAt(new Date(mealTime.getTime() + 90 * 60_000).toISOString(), response));
    }
    const result = computeFeatureImportance(baseInput({ glucose, meals }));
    expect(result).not.toBeNull();
    expect(result!.sampleSize).toBeGreaterThanOrEqual(30);
    expect(result!.coefficients[0]?.feature).toBe('lastMealCarbsG');
    expect(result!.coefficients[0]!.standardizedCoefficient).toBeGreaterThan(0);
    expect(result!.warning).toMatch(/not causes/);
  });

  it('handles empty input', () => {
    expect(computeFeatureImportance(baseInput())).toBeNull();
  });
});
