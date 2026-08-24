import { describe, expect, it } from 'vitest';
import {
  byHourOfDay,
  compareMorningVsEvening,
  gapsMinutes,
  histogram,
  summarizeGlucose,
  weekOverWeek,
} from '@/lib/analytics/glucose';
import type { GlucosePoint } from '@/lib/analytics/types';
import { DEFAULT_TARGET_RANGE } from '@/lib/domain/thresholds';

const TZ = 'America/Toronto';

function point(id: string, iso: string, valueMgdl: number, context: GlucosePoint['context'] = 'RANDOM'): GlucosePoint {
  return { id, takenAt: new Date(iso), valueMgdl, context };
}

describe('summarizeGlucose', () => {
  it('handles an empty reading list without throwing', () => {
    const summary = summarizeGlucose([], DEFAULT_TARGET_RANGE, TZ, new Date('2026-01-01T00:00:00Z'), new Date('2026-01-08T00:00:00Z'));
    expect(summary.count).toBe(0);
    expect(summary.averageMgdl).toBeNull();
    expect(summary.percentInRange).toBeNull();
    expect(summary.morningVsEvening).toBeNull();
  });

  it('handles a single reading', () => {
    const pts = [point('a', '2026-01-01T12:00:00Z', 120)];
    const summary = summarizeGlucose(pts, DEFAULT_TARGET_RANGE, TZ, new Date('2026-01-01T00:00:00Z'), new Date('2026-01-02T00:00:00Z'));
    expect(summary.count).toBe(1);
    expect(summary.averageMgdl).toBe(120);
    expect(summary.sdMgdl).toBeNull(); // n<2
    expect(summary.percentInRange).toBe(100);
  });

  it('computes percent in/below/above range as percentage of readings, not time', () => {
    const pts = [
      point('a', '2026-01-01T06:00:00Z', 60), // below (default low=70)
      point('b', '2026-01-01T07:00:00Z', 100), // in range
      point('c', '2026-01-01T08:00:00Z', 200), // above (default high=180)
      point('d', '2026-01-01T09:00:00Z', 150), // in range
    ];
    const summary = summarizeGlucose(pts, DEFAULT_TARGET_RANGE, TZ, new Date('2026-01-01T00:00:00Z'), new Date('2026-01-02T00:00:00Z'));
    expect(summary.percentBelowRange).toBe(25);
    expect(summary.percentInRange).toBe(50);
    expect(summary.percentAboveRange).toBe(25);
  });
});

describe('histogram', () => {
  it('buckets values into fixed-width bins with an overflow bucket', () => {
    const pts = [point('a', '2026-01-01T00:00:00Z', 45), point('b', '2026-01-01T00:00:00Z', 65), point('c', '2026-01-01T00:00:00Z', 405)];
    const buckets = histogram(pts, 20, 40, 400);
    const first = buckets.find((b) => b.from === 40);
    expect(first?.count).toBe(1);
    const overflow = buckets.find((b) => b.to === Infinity);
    expect(overflow?.count).toBe(1);
  });
});

describe('byHourOfDay', () => {
  it('groups readings by local hour, using the given timezone', () => {
    // 2026-01-01T05:00:00Z is 2026-01-01T00:00:00-05:00 in America/Toronto (EST, UTC-5)
    const pts = [point('a', '2026-01-01T05:00:00Z', 100)];
    const profile = byHourOfDay(pts, TZ);
    const hour0 = profile.find((h) => h.hour === 0);
    expect(hour0?.count).toBe(1);
    expect(hour0?.averageMgdl).toBe(100);
  });
});

describe('gapsMinutes', () => {
  it('computes consecutive gaps regardless of input order', () => {
    const pts = [
      point('b', '2026-01-01T01:00:00Z', 100),
      point('a', '2026-01-01T00:00:00Z', 100),
      point('c', '2026-01-01T01:30:00Z', 100),
    ];
    expect(gapsMinutes(pts)).toEqual([60, 30]);
  });
  it('returns an empty array for fewer than 2 readings', () => {
    expect(gapsMinutes([])).toEqual([]);
    expect(gapsMinutes([point('a', '2026-01-01T00:00:00Z', 100)])).toEqual([]);
  });
});

describe('compareMorningVsEvening', () => {
  it('returns null when either group has fewer than 2 readings', () => {
    const pts = [point('a', '2026-01-01T05:00:00Z', 100)]; // 00:00 local, morning bucket only
    expect(compareMorningVsEvening(pts, TZ)).toBeNull();
  });
  it('splits readings by local time-of-day bucket', () => {
    const pts = [
      // local 00:00, 01:00 -> overnight/morning
      point('a', '2026-01-01T05:00:00Z', 90),
      point('b', '2026-01-01T06:00:00Z', 95),
      // local 19:00, 20:00 -> evening
      point('c', '2026-01-01T00:00:00Z', 150),
      point('d', '2026-01-01T01:00:00Z', 160),
    ];
    const cmp = compareMorningVsEvening(pts, TZ);
    expect(cmp?.nA).toBe(2);
    expect(cmp?.nB).toBe(2);
    expect(cmp?.meanA).toBeCloseTo(92.5);
    expect(cmp?.meanB).toBeCloseTo(155);
  });
});

describe('weekOverWeek', () => {
  it('buckets readings into consecutive 7-day windows anchored to periodStart', () => {
    const periodStart = new Date('2026-01-01T00:00:00Z');
    const pts = [
      point('a', '2026-01-01T12:00:00Z', 100), // week 0
      point('b', '2026-01-03T12:00:00Z', 110), // week 0
      point('c', '2026-01-09T12:00:00Z', 120), // week 1
    ];
    const weeks = weekOverWeek(pts, 'UTC', periodStart);
    expect(weeks).toHaveLength(2);
    expect(weeks[0]?.count).toBe(2);
    expect(weeks[0]?.averageMgdl).toBeCloseTo(105);
    expect(weeks[1]?.count).toBe(1);
  });
});
