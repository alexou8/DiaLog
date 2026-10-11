/**
 * Input that used to be stored or parsed as-is and later blew up as a
 * RangeError (a 500, or a profile that broke every page that formats a date):
 * impossible calendar dates, unknown time zones and malformed locales.
 */
import { describe, expect, it } from 'vitest';
import {
  isRealLocalDateTime,
  isSupportedTimeZone,
  isValidLocale,
  localDateTimeSchema,
  onboardingSchema,
  preferencesSchema,
} from '@/lib/validation';
import { ALL_TIMEZONE_IDS } from '@/lib/timezones';
import { TIMEZONE_OPTIONS } from '@/app/app/settings/timezones';
import { LOCALES } from '@/lib/i18n/dictionaries';

describe('localDateTimeSchema', () => {
  it.each(['2026-03-14T08:30', '2026-03-14T08:30:59', '2024-02-29T00:00', '2026-12-31T23:59'])(
    'accepts the real instant %s',
    (value) => {
      expect(localDateTimeSchema.safeParse(value).success).toBe(true);
    },
  );

  it.each([
    '2025-13-45T99:99', // the review's example: every field out of range
    '2026-02-29T08:00', // not a leap year; Date would roll it to 1 March
    '2026-04-31T08:00', // April has 30 days
    '2026-00-10T08:00',
    '2026-01-00T08:00',
    '2026-01-10T24:00', // Date would roll it to the next day
    '2026-01-10T12:60',
    '2026-01-10T12:30:60',
    '2026-1-10T12:30',
    '2026-01-10 12:30',
    '',
  ])('rejects %s with a field error instead of a RangeError later', (value) => {
    const parsed = localDateTimeSchema.safeParse(value);
    expect(parsed.success).toBe(false);
    if (!parsed.success) {
      expect(parsed.error.issues[0]?.message).toBe('Please enter a valid date and time.');
    }
  });

  it('agrees with the schema in its exported predicate', () => {
    expect(isRealLocalDateTime('2024-02-29T12:00')).toBe(true);
    expect(isRealLocalDateTime('2023-02-29T12:00')).toBe(false);
  });
});

describe('isSupportedTimeZone', () => {
  it('accepts every zone the app itself offers', () => {
    for (const zone of [...ALL_TIMEZONE_IDS, ...TIMEZONE_OPTIONS.map((o) => o.value)]) {
      expect(isSupportedTimeZone(zone), zone).toBe(true);
    }
  });

  it('accepts UTC and both spellings of renamed zones browsers may report', () => {
    expect(isSupportedTimeZone('UTC')).toBe(true);
    expect(isSupportedTimeZone('Asia/Kolkata')).toBe(true);
    expect(isSupportedTimeZone('Asia/Calcutta')).toBe(true);
    expect(isSupportedTimeZone('America/Argentina/Buenos_Aires')).toBe(true);
  });

  it.each(['Foo/Bar', 'Mars/Olympus_Mons', 'America/', '../etc/passwd', '+05:00', 'UTC; DROP', ''])(
    'rejects %s',
    (zone) => {
      expect(isSupportedTimeZone(zone)).toBe(false);
    },
  );

  it('never accepts a value Intl would then throw on', () => {
    for (const zone of ['Foo/Bar', 'Nowhere/Special']) {
      expect(() => new Intl.DateTimeFormat('en', { timeZone: zone })).toThrow(RangeError);
      expect(isSupportedTimeZone(zone)).toBe(false);
    }
  });
});

describe('isValidLocale', () => {
  it('accepts every locale the app offers, and ordinary tags', () => {
    for (const locale of [...LOCALES, 'en', 'fr', 'en-US', 'zh-Hant-TW']) {
      expect(isValidLocale(locale), locale).toBe(true);
    }
  });

  it.each(['zz-!!', 'en_US_', '12', 'en--CA', '-'])('rejects %s', (locale) => {
    expect(isValidLocale(locale)).toBe(false);
  });
});

describe('profile schemas use the checks', () => {
  const preferences = {
    glucoseUnit: 'MGDL',
    locale: 'en-CA',
    timezone: 'America/Toronto',
    targetLow: '70',
    targetHigh: '180',
    detailLevel: 'STANDARD',
  };

  it('accepts a normal preferences form', () => {
    expect(preferencesSchema.safeParse(preferences).success).toBe(true);
  });

  it('rejects a crafted timezone or locale with a field error', () => {
    const tz = preferencesSchema.safeParse({ ...preferences, timezone: 'Foo/Bar' });
    expect(tz.success).toBe(false);
    if (!tz.success) expect(tz.error.issues[0]?.path).toEqual(['timezone']);

    const locale = preferencesSchema.safeParse({ ...preferences, locale: 'zz-!!' });
    expect(locale.success).toBe(false);
    if (!locale.success) expect(locale.error.issues[0]?.path).toEqual(['locale']);
  });

  it('applies to onboarding too', () => {
    const base = { condition: 'TYPE_2', glucoseUnit: 'MMOLL', goals: [] };
    expect(onboardingSchema.safeParse({ ...base, timezone: 'Europe/London' }).success).toBe(true);
    expect(onboardingSchema.safeParse({ ...base, timezone: 'Foo/Bar' }).success).toBe(false);
  });
});
