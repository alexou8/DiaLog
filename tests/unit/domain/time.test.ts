/**
 * lib/domain/time.ts converts every manually entered and every offset-less
 * imported timestamp. The old single-pass conversion sampled the zone offset
 * at the wrong instant, so for hours either side of a DST change readings
 * landed an hour off, and the repeated autumn hour collapsed two real
 * readings onto one dedupe key.
 */
import { describe, expect, it } from 'vitest';
import {
  dayKeyInZone,
  hourInZone,
  isValidTimeZone,
  startOfDayInZone,
  timeZoneOffsetMs,
  toLocalInputValue,
  weekdayInZone,
  zonedDateToUtc,
  zonedPartsToUtc,
} from '@/lib/domain/time';

const iso = (local: string, tz: string) => zonedDateToUtc(local, tz).toISOString();

describe('zonedDateToUtc across DST (northern hemisphere)', () => {
  const NY = 'America/New_York';

  it('resolves times after the spring-forward change with the new offset', () => {
    // 2024-03-10: clocks jump 02:00 EST -> 03:00 EDT (07:00Z).
    expect(iso('2024-03-10T03:30:00', NY)).toBe('2024-03-10T07:30:00.000Z');
    expect(iso('2024-03-10T05:00:00', NY)).toBe('2024-03-10T09:00:00.000Z');
  });

  it('resolves times before the spring-forward change with the old offset', () => {
    expect(iso('2024-03-10T01:30:00', NY)).toBe('2024-03-10T06:30:00.000Z');
  });

  it('shifts a nonexistent (skipped) time forward across the gap', () => {
    // 02:30 never happens that day; it is read as 03:30 EDT.
    expect(iso('2024-03-10T02:30:00', NY)).toBe('2024-03-10T07:30:00.000Z');
  });

  it('resolves the repeated autumn hour to its first (earlier) occurrence', () => {
    // 2024-11-03: 01:00-02:00 happens twice, first EDT (UTC-4) then EST (UTC-5).
    expect(iso('2024-11-03T01:30:00', NY)).toBe('2024-11-03T05:30:00.000Z');
  });

  it('resolves times after the fall-back change with the new offset', () => {
    expect(iso('2024-11-03T02:30:00', NY)).toBe('2024-11-03T07:30:00.000Z');
    expect(iso('2026-11-01T03:00:00', 'America/Toronto')).toBe('2026-11-01T08:00:00.000Z');
  });
});

describe('zonedDateToUtc across DST (southern hemisphere)', () => {
  const SYD = 'Australia/Sydney';

  it('resolves the repeated autumn hour (April) to its earlier occurrence', () => {
    // 2026-04-05 03:00 AEDT -> 02:00 AEST; 02:30 happens at UTC+11 first.
    expect(iso('2026-04-05T02:30:00', SYD)).toBe('2026-04-04T15:30:00.000Z');
    expect(iso('2026-04-05T04:00:00', SYD)).toBe('2026-04-04T18:00:00.000Z');
  });

  it('shifts the skipped spring hour (October) forward', () => {
    // 2026-10-04 02:00 AEST -> 03:00 AEDT; 02:30 is read as 03:30 AEDT.
    expect(iso('2026-10-04T02:30:00', SYD)).toBe('2026-10-03T16:30:00.000Z');
  });

  it('starts the local day at local midnight on a transition day', () => {
    const start = startOfDayInZone(new Date('2026-04-05T12:00:00Z'), SYD);
    expect(start.toISOString()).toBe('2026-04-04T13:00:00.000Z'); // 00:00 AEDT
    expect(dayKeyInZone(start, SYD)).toBe('2026-04-05');
    const auckland = startOfDayInZone(new Date('2026-04-05T12:00:00Z'), 'Pacific/Auckland');
    expect(toLocalInputValue(auckland, 'Pacific/Auckland')).toBe('2026-04-06T00:00');
  });
});

describe('zonedDateToUtc in fixed-offset zones', () => {
  it('is the identity for UTC', () => {
    expect(iso('2026-06-01T12:00:00', 'UTC')).toBe('2026-06-01T12:00:00.000Z');
    expect(iso('2026-06-01T12:00:00', 'Etc/UTC')).toBe('2026-06-01T12:00:00.000Z');
  });

  it('handles a half-hour offset', () => {
    expect(iso('2026-06-01T12:00:00', 'Asia/Kolkata')).toBe('2026-06-01T06:30:00.000Z');
  });

  it('agrees with zonedPartsToUtc', () => {
    expect(zonedPartsToUtc(2024, 3, 10, 3, 30, 0, 'America/New_York').toISOString()).toBe(
      '2024-03-10T07:30:00.000Z',
    );
  });

  it('throws on an invalid local date-time instead of storing Invalid Date', () => {
    expect(() => zonedDateToUtc('2025-13-45T99:99', 'UTC')).toThrow(RangeError);
  });
});

describe('zone helpers', () => {
  it('reports offsets ignoring sub-second remainders', () => {
    expect(timeZoneOffsetMs(new Date('2026-01-10T12:00:00.750Z'), 'America/Toronto')).toBe(
      -5 * 3_600_000,
    );
  });

  it('reads hour, weekday and day key in the zone', () => {
    const d = new Date('2026-01-04T03:30:00Z'); // Sat 22:30 in New York
    expect(hourInZone(d, 'America/New_York')).toBe(22);
    expect(weekdayInZone(d, 'America/New_York')).toBe(6);
    expect(dayKeyInZone(d, 'America/New_York')).toBe('2026-01-03');
    expect(hourInZone(new Date('2026-01-04T00:00:00Z'), 'UTC')).toBe(0);
  });

  it('validates zone names', () => {
    expect(isValidTimeZone('Europe/Paris')).toBe(true);
    expect(isValidTimeZone('Not/AZone')).toBe(false);
  });
});

describe('performance', () => {
  it('converts 10k timestamps well under a second (formatters are cached)', () => {
    // Building an Intl.DateTimeFormat per call cost ~50 µs each, which made
    // a 90-day CGM analysis spend seconds in formatter construction.
    const start = performance.now();
    for (let i = 0; i < 10_000; i++) {
      const d = new Date(Date.UTC(2026, 0, 1) + i * 5 * 60_000);
      zonedDateToUtc('2026-03-08T03:30:00', 'America/New_York');
      dayKeyInZone(d, 'America/New_York');
      hourInZone(d, 'Europe/London');
    }
    expect(performance.now() - start).toBeLessThan(1_000);
  });
});
