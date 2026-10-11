import { describe, expect, it } from 'vitest';
import {
  fastingWeekdayVsWeekend,
  postDinnerActivityComparison,
  postMealResponseByCarbBucket,
  runAssociations,
  sleepDurationComparison,
  stressComparison,
} from '@/lib/analytics/associations';
import type {
  AnalyticsInput,
  ExercisePoint,
  GlucosePoint,
  MealPoint,
  MoodPoint,
  SleepPoint,
} from '@/lib/analytics/types';
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
    periodEnd: new Date('2026-03-01T00:00:00Z'),
    ...overrides,
  };
}

function glucoseAt(
  iso: string,
  valueMgdl: number,
  context: GlucosePoint['context'] = 'RANDOM',
): GlucosePoint {
  return { id: iso + Math.random(), takenAt: new Date(iso), valueMgdl, context };
}

describe('postMealResponseByCarbBucket', () => {
  it('returns null when there is not enough data', () => {
    const input = baseInput({
      meals: [
        {
          id: 'm1',
          takenAt: new Date('2026-01-01T12:00:00Z'),
          mealType: 'LUNCH',
          carbsG: 20,
          description: '',
        },
      ],
      glucose: [glucoseAt('2026-01-01T13:00:00Z', 100)],
    });
    expect(postMealResponseByCarbBucket(input)).toBeNull();
  });

  it('finds a Finding with a higher average for high-carb meals than low-carb meals', () => {
    const meals: MealPoint[] = [];
    const glucose: GlucosePoint[] = [];
    // 12 low-carb meals -> modest post-meal rise; 12 high-carb meals -> bigger rise
    for (let i = 0; i < 12; i++) {
      const mealTime = new Date(Date.UTC(2026, 0, i + 1, 12, 0, 0));
      meals.push({
        id: `low-${i}`,
        takenAt: mealTime,
        mealType: 'LUNCH',
        carbsG: 15,
        description: '',
      });
      glucose.push(
        glucoseAt(new Date(mealTime.getTime() + 90 * 60_000).toISOString(), 110 + (i % 3)),
      );

      const highMealTime = new Date(Date.UTC(2026, 1, i + 1, 12, 0, 0));
      meals.push({
        id: `high-${i}`,
        takenAt: highMealTime,
        mealType: 'LUNCH',
        carbsG: 90,
        description: '',
      });
      glucose.push(
        glucoseAt(new Date(highMealTime.getTime() + 90 * 60_000).toISOString(), 190 + (i % 3)),
      );
    }
    const input = baseInput({ meals, glucose });
    const finding = postMealResponseByCarbBucket(input);
    expect(finding).not.toBeNull();
    expect(finding!.sampleSize).toBe(24);
    expect(finding!.metrics.highAvgMgdl as number).toBeGreaterThan(
      finding!.metrics.lowAvgMgdl as number,
    );
    expect(finding!.evidenceLevel).not.toBe('INSUFFICIENT');
  });
});

describe('postDinnerActivityComparison', () => {
  it('returns null with too few dinner days', () => {
    const input = baseInput();
    expect(postDinnerActivityComparison(input)).toBeNull();
  });

  it('compares post-dinner glucose on days with vs without logged activity', () => {
    const meals: MealPoint[] = [];
    const glucose: GlucosePoint[] = [];
    const exercise: ExercisePoint[] = [];
    for (let day = 1; day <= 10; day++) {
      const dinnerTime = new Date(Date.UTC(2026, 0, day, 18, 0, 0));
      meals.push({
        id: `dinner-${day}`,
        takenAt: dinnerTime,
        mealType: 'DINNER',
        carbsG: 50,
        description: '',
      });
      const postReading = new Date(dinnerTime.getTime() + 90 * 60_000);
      const active = day % 2 === 0;
      glucose.push(glucoseAt(postReading.toISOString(), active ? 120 : 170));
      if (active) {
        exercise.push({
          id: `walk-${day}`,
          takenAt: new Date(dinnerTime.getTime() + 30 * 60_000),
          endedAt: null,
          durationMin: 20,
          activity: 'walk',
          intensity: 'LIGHT',
        });
      }
    }
    const input = baseInput({ meals, glucose, exercise });
    const finding = postDinnerActivityComparison(input);
    expect(finding).not.toBeNull();
    expect(finding!.metrics.avgWithActivityMgdl as number).toBeLessThan(
      finding!.metrics.avgWithoutActivityMgdl as number,
    );
  });
});

describe('sleepDurationComparison', () => {
  it('returns null without enough sleep sessions', () => {
    expect(sleepDurationComparison(baseInput())).toBeNull();
  });

  it('compares morning glucose after short vs adequate sleep', () => {
    const sleep: SleepPoint[] = [];
    const glucose: GlucosePoint[] = [];
    for (let day = 1; day <= 10; day++) {
      const wake = new Date(Date.UTC(2026, 0, day, 7, 0, 0));
      const short = day % 2 === 0;
      sleep.push({
        id: `sleep-${day}`,
        takenAt: new Date(wake.getTime() - (short ? 5 : 8) * 60 * 60_000),
        endedAt: wake,
        durationMin: (short ? 5 : 8) * 60,
        quality: null,
      });
      glucose.push(
        glucoseAt(new Date(wake.getTime() + 60 * 60_000).toISOString(), short ? 160 : 110),
      );
    }
    const finding = sleepDurationComparison(baseInput({ sleep, glucose }));
    expect(finding).not.toBeNull();
    expect(finding!.metrics.avgAfterShortSleepMgdl as number).toBeGreaterThan(
      finding!.metrics.avgAfterAdequateSleepMgdl as number,
    );
  });
});

describe('fastingWeekdayVsWeekend', () => {
  it('returns null without enough fasting readings', () => {
    expect(fastingWeekdayVsWeekend(baseInput())).toBeNull();
  });

  it('splits fasting readings by local weekday vs weekend', () => {
    const glucose: GlucosePoint[] = [];
    // 2026-01-01 is a Thursday; 2026-01-03/04 are Sat/Sun
    for (let i = 0; i < 8; i++) {
      glucose.push(
        glucoseAt(new Date(Date.UTC(2026, 0, 5 + i * 7, 7, 0, 0)).toISOString(), 100, 'FASTING'),
      ); // Mondays
      glucose.push(
        glucoseAt(new Date(Date.UTC(2026, 0, 3 + i * 7, 7, 0, 0)).toISOString(), 130, 'FASTING'),
      ); // Saturdays
    }
    const finding = fastingWeekdayVsWeekend(baseInput({ glucose }));
    expect(finding).not.toBeNull();
    expect(finding!.metrics.avgWeekendMgdl as number).toBeGreaterThan(
      finding!.metrics.avgWeekdayMgdl as number,
    );
  });
});

describe('stressComparison', () => {
  it('returns null without enough mood entries', () => {
    expect(stressComparison(baseInput())).toBeNull();
  });

  it('compares average daily glucose on high vs low stress days', () => {
    const moods: MoodPoint[] = [];
    const glucose: GlucosePoint[] = [];
    for (let day = 1; day <= 10; day++) {
      const dayStart = new Date(Date.UTC(2026, 0, day, 10, 0, 0));
      const highStress = day % 2 === 0;
      moods.push({ id: `mood-${day}`, takenAt: dayStart, mood: 3, stress: highStress ? 5 : 1 });
      glucose.push(glucoseAt(dayStart.toISOString(), highStress ? 160 : 105));
    }
    const finding = stressComparison(baseInput({ moods, glucose }));
    expect(finding).not.toBeNull();
    expect(finding!.metrics.avgHighStressMgdl as number).toBeGreaterThan(
      finding!.metrics.avgLowStressMgdl as number,
    );
  });
});

describe('runAssociations', () => {
  it('returns an empty array for empty input', () => {
    expect(runAssociations(baseInput())).toEqual([]);
  });
});

describe('evidence grading of comparisons', () => {
  /** One short-sleep or adequate-sleep night per day, with a morning reading. */
  function sleepInput(nights: { short: boolean; morning: number }[]): AnalyticsInput {
    const sleep: SleepPoint[] = [];
    const glucose: GlucosePoint[] = [];
    nights.forEach(({ short, morning }, i) => {
      const wake = new Date(Date.UTC(2026, 0, 1 + i, 7, 0, 0));
      const hours = short ? 5 : 8;
      sleep.push({
        id: `s-${i}`,
        takenAt: new Date(wake.getTime() - hours * 3_600_000),
        endedAt: wake,
        durationMin: hours * 60,
        quality: null,
      });
      glucose.push(glucoseAt(new Date(wake.getTime() + 3_600_000).toISOString(), morning));
    });
    return baseInput({ sleep, glucose });
  }

  it('does not call a lopsided comparison consistent', () => {
    // 38 adequate nights vs 2 short: 40 total used to grade CONSISTENT.
    const nights = [
      ...Array.from({ length: 38 }, (_, i) => ({ short: false, morning: 100 + (i % 5) })),
      { short: true, morning: 160 },
      { short: true, morning: 170 },
    ];
    expect(sleepDurationComparison(sleepInput(nights))).toBeNull();
  });

  it('caps a large but noise-level difference at an early signal', () => {
    // 25 vs 25 nights with identical distributions: p ~ 1.
    const nights = Array.from({ length: 50 }, (_, i) => ({
      short: i % 2 === 0,
      morning: 100 + (Math.floor(i / 2) % 10),
    }));
    const finding = sleepDurationComparison(sleepInput(nights));
    expect(finding).not.toBeNull();
    expect(finding!.sampleSize).toBe(50);
    expect(finding!.metrics.welchPValue as number).toBeGreaterThan(0.05);
    expect(finding!.evidenceLevel).toBe('EARLY');
  });

  it('grades a balanced, clear difference as consistent and reports its confidence interval', () => {
    const nights = Array.from({ length: 50 }, (_, i) => ({
      short: i % 2 === 0,
      morning: (i % 2 === 0 ? 150 : 110) + (Math.floor(i / 2) % 7),
    }));
    const finding = sleepDurationComparison(sleepInput(nights));
    expect(finding!.evidenceLevel).toBe('CONSISTENT');
    const low = finding!.metrics.differenceCiLowMgdl as number;
    const high = finding!.metrics.differenceCiHighMgdl as number;
    expect(low).toBeGreaterThan(0);
    expect(high).toBeGreaterThan(low);
    expect(40).toBeGreaterThan(low);
    expect(40).toBeLessThan(high);
  });

  it('never loses a finding when more high-carb meals are logged', () => {
    // 20 low vs 1 high is described (EARLY); 20 vs 3 is compared, and the
    // comparison is graded on its smaller arm (2 × 3 = 6, below the early
    // threshold). That used to return null, so logging more removed a finding.
    function carbInput(lowN: number, highN: number): AnalyticsInput {
      const meals: MealPoint[] = [];
      const glucose: GlucosePoint[] = [];
      const add = (id: string, at: Date, carbsG: number, value: number) => {
        meals.push({ id, takenAt: at, mealType: 'LUNCH', carbsG, description: '' });
        glucose.push(glucoseAt(new Date(at.getTime() + 90 * 60_000).toISOString(), value));
      };
      for (let i = 0; i < lowN; i++) add(`l${i}`, new Date(Date.UTC(2026, 0, i + 1, 12)), 15, 110);
      for (let i = 0; i < highN; i++) add(`h${i}`, new Date(Date.UTC(2026, 1, i + 1, 12)), 90, 190);
      return baseInput({ meals, glucose });
    }

    let previous: ReturnType<typeof postMealResponseByCarbBucket> = null;
    for (let highN = 1; highN <= 12; highN++) {
      const finding = postMealResponseByCarbBucket(carbInput(20, highN));
      if (previous !== null) expect(finding).not.toBeNull();
      previous = finding;
    }
    expect(postMealResponseByCarbBucket(carbInput(20, 1))?.evidenceLevel).toBe('EARLY');
    expect(postMealResponseByCarbBucket(carbInput(20, 3))?.evidenceLevel).toBe('EARLY');
  });

  it('finds post-meal readings correctly when glucose is not in time order', () => {
    const meals: MealPoint[] = [];
    const glucose: GlucosePoint[] = [];
    for (let i = 0; i < 12; i++) {
      const low = new Date(Date.UTC(2026, 0, i + 1, 12));
      const high = new Date(Date.UTC(2026, 0, i + 1, 18));
      meals.push({ id: `l${i}`, takenAt: low, mealType: 'LUNCH', carbsG: 15, description: '' });
      meals.push({ id: `h${i}`, takenAt: high, mealType: 'DINNER', carbsG: 90, description: '' });
      // Out of order on purpose, plus readings just outside the 60-180 min window.
      glucose.unshift(glucoseAt(new Date(high.getTime() + 120 * 60_000).toISOString(), 200 + i));
      glucose.push(glucoseAt(new Date(low.getTime() + 90 * 60_000).toISOString(), 110 + i));
      glucose.push(glucoseAt(new Date(low.getTime() + 59 * 60_000).toISOString(), 400));
      glucose.push(glucoseAt(new Date(low.getTime() + 181 * 60_000).toISOString(), 400));
    }
    const finding = postMealResponseByCarbBucket(baseInput({ meals, glucose }));
    expect(finding!.metrics.lowAvgMgdl).toBeCloseTo(115.5, 5);
    expect(finding!.metrics.highAvgMgdl).toBeCloseTo(205.5, 5);
  });
});
