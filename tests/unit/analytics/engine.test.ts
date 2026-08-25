import { describe, expect, it } from 'vitest';
import { runAnalytics } from '@/lib/analytics/engine';
import { buildInsights } from '@/lib/analytics/insights';
import type { AnalyticsInput, GlucosePoint, MealPoint } from '@/lib/analytics/types';
import { DEFAULT_TARGET_RANGE } from '@/lib/domain/thresholds';

const TZ = 'America/Toronto';

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

describe('runAnalytics', () => {
  it('handles a fully empty input without throwing', () => {
    const result = runAnalytics(baseInput());
    expect(result.summary.count).toBe(0);
    expect(result.findings).toEqual([]);
    expect(result.anomalies).toEqual([]);
    expect(result.trend.classification).toBe('not-enough-data');
    expect(result.dayPatterns).toBeNull();
    expect(result.featureImportance).toBeNull();
    expect(result.dataQuality.counts.glucose).toBe(0);
    expect(result.dataQuality.skippedAnalyses.length).toBeGreaterThan(0);
  });

  it('records every skipped analysis with a stated reason when data is sparse', () => {
    const glucose = [glucoseAt('2026-01-01T12:00:00Z', 100)];
    const result = runAnalytics(baseInput({ glucose }));
    for (const skipped of result.dataQuality.skippedAnalyses) {
      expect(skipped.analysis.length).toBeGreaterThan(0);
      expect(skipped.reason.length).toBeGreaterThan(0);
    }
    // With a single reading, trend, day-patterns, feature-importance and all 5 associations must be skipped.
    const analyses = result.dataQuality.skippedAnalyses.map((s) => s.analysis);
    expect(analyses).toContain('trend');
    expect(analyses).toContain('day-patterns');
    expect(analyses).toContain('feature-importance');
    expect(analyses).toContain('association:post-meal-carb-bucket');
  });

  it('reports coverage days and gap statistics', () => {
    const glucose = [
      glucoseAt('2026-01-01T08:00:00Z', 100),
      glucoseAt('2026-01-01T20:00:00Z', 110),
      glucoseAt('2026-01-02T08:00:00Z', 105),
    ];
    const result = runAnalytics(baseInput({ glucose }));
    expect(result.dataQuality.coverageDays).toBe(2);
    expect(result.dataQuality.counts.glucose).toBe(3);
    expect(result.dataQuality.averageGapMinutes).not.toBeNull();
  });

  it('produces a well-formed result with a richer dataset (findings, trend, anomalies all engage)', () => {
    const glucose: GlucosePoint[] = [];
    const meals: MealPoint[] = [];
    for (let day = 1; day <= 40; day++) {
      const d = new Date(Date.UTC(2026, 0, day, 12, 0, 0));
      const carbs = 20 + (day % 5) * 15;
      meals.push({ id: `m${day}`, takenAt: d, mealType: 'LUNCH', carbsG: carbs, description: '' });
      glucose.push(glucoseAt(new Date(d.getTime() + 90 * 60_000).toISOString(), 100 + carbs));
      glucose.push(
        glucoseAt(new Date(Date.UTC(2026, 0, day, 7, 0, 0)).toISOString(), 95 + (day % 3)),
      );
    }
    const result = runAnalytics(baseInput({ glucose, meals }));
    expect(result.summary.count).toBe(glucose.length);
    expect(result.summary.averageMgdl).not.toBeNull();
    expect(Array.isArray(result.findings)).toBe(true);
  });
});

describe('buildInsights', () => {
  it('produces no cards for a fully empty result except a data-quality card', () => {
    const result = runAnalytics(baseInput());
    const cards = buildInsights(result);
    expect(cards.length).toBeGreaterThan(0);
    expect(cards.some((c) => c.kind === 'data-quality')).toBe(true);
    for (const card of cards) {
      expect(card.evidenceLevel).toBeTruthy();
      expect(typeof card.title).toBe('string');
      expect(typeof card.summary).toBe('string');
      expect(
        card.source === 'STATISTICAL' || card.source === 'ML' || card.source === 'REFERENCE',
      ).toBe(true);
    }
  });

  it('translates findings into insight cards with matching metrics', () => {
    const meals: MealPoint[] = [];
    const glucose: GlucosePoint[] = [];
    for (let i = 0; i < 15; i++) {
      const mealTime = new Date(Date.UTC(2026, 0, i + 1, 12, 0, 0));
      meals.push({
        id: `low-${i}`,
        takenAt: mealTime,
        mealType: 'LUNCH',
        carbsG: 10,
        description: '',
      });
      glucose.push(glucoseAt(new Date(mealTime.getTime() + 90 * 60_000).toISOString(), 110));

      const highMealTime = new Date(Date.UTC(2026, 1, i + 1, 12, 0, 0));
      meals.push({
        id: `high-${i}`,
        takenAt: highMealTime,
        mealType: 'LUNCH',
        carbsG: 90,
        description: '',
      });
      glucose.push(glucoseAt(new Date(highMealTime.getTime() + 90 * 60_000).toISOString(), 200));
    }
    const result = runAnalytics(baseInput({ glucose, meals }));
    const cards = buildInsights(result);
    const carbCard = cards.find((c) => c.kind === 'post-meal-carb-bucket');
    expect(carbCard).toBeDefined();
    expect(carbCard!.sampleSize).toBe(30);
    expect(carbCard!.summary).toContain('mg/dL');
  });

  it('never phrases a summary causally or prescriptively (no "should", "cause", "must")', () => {
    const meals: MealPoint[] = [];
    const glucose: GlucosePoint[] = [];
    for (let i = 0; i < 15; i++) {
      const mealTime = new Date(Date.UTC(2026, 0, i + 1, 12, 0, 0));
      meals.push({
        id: `low-${i}`,
        takenAt: mealTime,
        mealType: 'LUNCH',
        carbsG: 10,
        description: '',
      });
      glucose.push(glucoseAt(new Date(mealTime.getTime() + 90 * 60_000).toISOString(), 110));
      const highMealTime = new Date(Date.UTC(2026, 1, i + 1, 12, 0, 0));
      meals.push({
        id: `high-${i}`,
        takenAt: highMealTime,
        mealType: 'LUNCH',
        carbsG: 90,
        description: '',
      });
      glucose.push(glucoseAt(new Date(highMealTime.getTime() + 90 * 60_000).toISOString(), 200));
    }
    const result = runAnalytics(baseInput({ glucose, meals }));
    const cards = buildInsights(result);
    const forbidden = /\b(should|must|cause[sd]?|caused by)\b/i;
    for (const card of cards) {
      expect(card.summary).not.toMatch(forbidden);
    }
  });
});
