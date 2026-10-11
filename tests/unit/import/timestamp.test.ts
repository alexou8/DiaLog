import { describe, expect, it } from 'vitest';
import { parseTimestamp, resolveDateOrder, tzOffsetMinutes } from '@/lib/import/parse';

const NOW = new Date('2026-08-24T12:00:00Z');

describe('parseTimestamp', () => {
  it('parses ISO 8601 with Z offset', () => {
    const r = parseTimestamp('2026-01-10T08:10:00Z', { now: NOW });
    expect(r.date?.toISOString()).toBe('2026-01-10T08:10:00.000Z');
    expect(r.error).toBeUndefined();
  });

  it('parses ISO 8601 with explicit numeric offset', () => {
    const r = parseTimestamp('2026-01-10T08:10:00-05:00', { now: NOW });
    expect(r.date?.toISOString()).toBe('2026-01-10T13:10:00.000Z');
  });

  it('parses "YYYY-MM-DD HH:mm:ss" with a space separator, UTC timezone', () => {
    const r = parseTimestamp('2026-01-10 08:10:00', { timezone: 'UTC', now: NOW });
    expect(r.date?.toISOString()).toBe('2026-01-10T08:10:00.000Z');
  });

  it('parses "YYYY-MM-DD HH:mm" without seconds', () => {
    const r = parseTimestamp('2026-01-10 08:10', { timezone: 'UTC', now: NOW });
    expect(r.date?.toISOString()).toBe('2026-01-10T08:10:00.000Z');
  });

  it('interprets a naive timestamp in a named IANA timezone', () => {
    const r = parseTimestamp('2026-01-10 08:00:00', { timezone: 'America/Toronto', now: NOW });
    // Toronto is UTC-5 in January (standard time).
    expect(r.date?.toISOString()).toBe('2026-01-10T13:00:00.000Z');
  });

  it('parses DD/MM/YYYY HH:mm when dateOrder is DMY', () => {
    const r = parseTimestamp('20/01/2026 08:00', { timezone: 'UTC', dateOrder: 'DMY', now: NOW });
    expect(r.date?.toISOString()).toBe('2026-01-20T08:00:00.000Z');
  });

  it('parses MM/DD/YYYY hh:mm AM/PM when dateOrder is MDY', () => {
    const r = parseTimestamp('01/20/2026 08:00 PM', {
      timezone: 'UTC',
      dateOrder: 'MDY',
      now: NOW,
    });
    expect(r.date?.toISOString()).toBe('2026-01-20T20:00:00.000Z');
  });

  it('handles 12 AM / 12 PM edge cases', () => {
    const midnight = parseTimestamp('01/20/2026 12:00 AM', {
      timezone: 'UTC',
      dateOrder: 'MDY',
      now: NOW,
    });
    expect(midnight.date?.toISOString()).toBe('2026-01-20T00:00:00.000Z');
    const noon = parseTimestamp('01/20/2026 12:00 PM', {
      timezone: 'UTC',
      dateOrder: 'MDY',
      now: NOW,
    });
    expect(noon.date?.toISOString()).toBe('2026-01-20T12:00:00.000Z');
  });

  it('self-corrects an impossible month when the assumed order is wrong', () => {
    // day=20 can't be a month, so even under MDY assumption this must be DMY.
    const r = parseTimestamp('20/01/2026 08:00', { timezone: 'UTC', dateOrder: 'MDY', now: NOW });
    expect(r.date?.toISOString()).toBe('2026-01-20T08:00:00.000Z');
  });

  it('parses Excel serial dates', () => {
    // 45678 -> 2025-01-01 in Excel's serial scheme.
    const r = parseTimestamp(45678, { now: NOW });
    expect(r.date).not.toBeNull();
    expect(r.date?.getUTCFullYear()).toBe(2025);
  });

  it('parses epoch seconds', () => {
    const r = parseTimestamp(1768032000, { now: NOW }); // 2026-01-10T08:00:00Z
    expect(r.date?.toISOString()).toBe('2026-01-10T08:00:00.000Z');
  });

  it('parses epoch milliseconds', () => {
    const r = parseTimestamp(1768032000000, { now: NOW });
    expect(r.date?.toISOString()).toBe('2026-01-10T08:00:00.000Z');
  });

  it('parses a numeric-looking string as epoch millis', () => {
    const r = parseTimestamp('1768032000000', { now: NOW });
    expect(r.date?.toISOString()).toBe('2026-01-10T08:00:00.000Z');
  });

  it('flags a missing timestamp', () => {
    expect(parseTimestamp(null, { now: NOW }).error).toBe('MISSING_TIMESTAMP');
    expect(parseTimestamp(undefined, { now: NOW }).error).toBe('MISSING_TIMESTAMP');
    expect(parseTimestamp('', { now: NOW }).error).toBe('MISSING_TIMESTAMP');
    expect(parseTimestamp('   ', { now: NOW }).error).toBe('MISSING_TIMESTAMP');
  });

  it('flags an unparseable timestamp', () => {
    const r = parseTimestamp('not a date', { now: NOW });
    expect(r.error).toBe('INVALID_TIMESTAMP');
    expect(r.date).toBeNull();
  });

  it('rejects timestamps more than 24h in the future', () => {
    const future = new Date(NOW.getTime() + 48 * 3600 * 1000).toISOString();
    const r = parseTimestamp(future, { now: NOW });
    expect(r.error).toBe('FUTURE_TIMESTAMP');
    expect(r.date).not.toBeNull(); // date is still returned for context, just flagged
  });

  it('accepts a timestamp just under the 24h future window', () => {
    const soon = new Date(NOW.getTime() + 23 * 3600 * 1000).toISOString();
    const r = parseTimestamp(soon, { now: NOW });
    expect(r.error).toBeUndefined();
  });
});

describe('resolveDateOrder', () => {
  it('infers day-first when a component exceeds 12', () => {
    const { order, assumed } = resolveDateOrder(['20/01/2026 08:00', '05/02/2026 09:00']);
    expect(order).toBe('DMY');
    expect(assumed).toBe(false);
  });

  it('infers month-first when the second component exceeds 12', () => {
    const { order, assumed } = resolveDateOrder(['01/20/2026 08:00']);
    expect(order).toBe('MDY');
    expect(assumed).toBe(false);
  });

  it('falls back to an explicit dateOrder when genuinely ambiguous', () => {
    const { order, assumed } = resolveDateOrder(['01/02/2026 08:00', '03/04/2026 08:00'], 'DMY');
    expect(order).toBe('DMY');
    expect(assumed).toBe(false);
  });

  it('assumes MDY with a flag when ambiguous and no explicit order given', () => {
    const { order, assumed } = resolveDateOrder(['01/02/2026 08:00', '03/04/2026 08:00']);
    expect(order).toBe('MDY');
    expect(assumed).toBe(true);
  });
});

describe('tzOffsetMinutes', () => {
  it('returns 0 for UTC', () => {
    expect(tzOffsetMinutes(new Date('2026-01-10T00:00:00Z'), 'UTC')).toBe(0);
  });

  it('returns a negative offset for a Western-hemisphere zone in winter', () => {
    const offset = tzOffsetMinutes(new Date('2026-01-10T12:00:00Z'), 'America/Toronto');
    expect(offset).toBe(-300); // EST, UTC-5
  });
});

describe('local timestamps across DST (shared lib/domain/time helper)', () => {
  const NOW = new Date('2026-12-31T00:00:00Z');

  it('uses the post-transition offset after spring-forward', () => {
    const r = parseTimestamp('2026-03-08 05:00', { timezone: 'America/Toronto', now: NOW });
    expect(r.date?.toISOString()).toBe('2026-03-08T09:00:00.000Z');
  });

  it('keeps the two readings of the repeated autumn hour distinct from the next hour', () => {
    const first = parseTimestamp('2026-11-01 01:30', { timezone: 'America/Toronto', now: NOW });
    const after = parseTimestamp('2026-11-01 03:00', { timezone: 'America/Toronto', now: NOW });
    expect(first.date?.toISOString()).toBe('2026-11-01T05:30:00.000Z');
    expect(after.date?.toISOString()).toBe('2026-11-01T08:00:00.000Z');
  });

  it('throws for an unknown zone rather than silently using UTC', () => {
    expect(() => tzOffsetMinutes(new Date('2026-01-10T00:00:00Z'), 'Not/AZone')).toThrow(
      RangeError,
    );
  });
});

describe('calendar-impossible timestamps', () => {
  const NOW = new Date('2026-12-31T00:00:00Z');

  it('rejects dates that Date.UTC would roll over', () => {
    for (const raw of ['2026-02-30 08:00', '2026-13-01 08:00', '2026-01-10 24:30', '2026-02-29']) {
      expect(parseTimestamp(raw, { now: NOW }).error).toBe('INVALID_TIMESTAMP');
    }
    expect(parseTimestamp('31/04/2026 08:00', { dateOrder: 'DMY', now: NOW }).error).toBe(
      'INVALID_TIMESTAMP',
    );
    expect(parseTimestamp('2026-02-30T08:00:00Z', { now: NOW }).error).toBe('INVALID_TIMESTAMP');
  });

  it('still accepts a real leap day', () => {
    expect(parseTimestamp('2028-02-29 08:00', { now: new Date('2028-03-01') }).date).not.toBeNull();
  });
});
