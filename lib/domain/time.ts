/** Locale- and timezone-aware date helpers. All storage is UTC. */

const MINUTE_MS = 60_000;
const DAY_MS = 86_400_000;

/**
 * `Intl.DateTimeFormat` construction costs ~50 µs (it loads ICU zone data),
 * while `format()` on an existing instance costs ~1 µs. Analytics calls these
 * helpers about a dozen times per reading, so building a formatter per call
 * spent ~15 s of a 90-day CGM analysis in construction alone. Instances are
 * stateless and safe to share, so they are cached per (kind, zone, locale).
 *
 * The cap only exists so a stream of distinct zone strings cannot grow the
 * map without bound; real traffic uses a handful of zones.
 */
const FORMATTER_CACHE_LIMIT = 512;
const formatterCache = new Map<string, Intl.DateTimeFormat>();

type FormatterKind = 'parts' | 'hour' | 'weekday' | 'day' | 'input';

const FORMATTER_OPTIONS: Record<FormatterKind, Intl.DateTimeFormatOptions> = {
  // `hourCycle: 'h23'` rather than `hour12: false`: the latter renders
  // midnight as "24" in some ICU versions.
  parts: {
    hourCycle: 'h23',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  },
  hour: { hour: '2-digit', hourCycle: 'h23' },
  weekday: { weekday: 'short' },
  day: { year: 'numeric', month: '2-digit', day: '2-digit' },
  input: {
    hourCycle: 'h23',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  },
};

/** `en-CA` formats dates as YYYY-MM-DD; `en-US` gives predictable part names. */
const FORMATTER_LOCALE: Record<FormatterKind, string> = {
  parts: 'en-US',
  hour: 'en-US',
  weekday: 'en-US',
  day: 'en-CA',
  input: 'en-CA',
};

function formatterFor(kind: FormatterKind, timeZone: string): Intl.DateTimeFormat {
  const key = `${kind}|${timeZone}`;
  let dtf = formatterCache.get(key);
  if (!dtf) {
    // Throws RangeError for an unknown zone — nothing is cached in that case.
    dtf = new Intl.DateTimeFormat(FORMATTER_LOCALE[kind], {
      ...FORMATTER_OPTIONS[kind],
      timeZone,
    });
    if (formatterCache.size >= FORMATTER_CACHE_LIMIT) formatterCache.clear();
    formatterCache.set(key, dtf);
  }
  return dtf;
}

/** True when `timeZone` is an IANA zone this runtime's Intl data recognises. */
export function isValidTimeZone(timeZone: string): boolean {
  try {
    formatterFor('parts', timeZone);
    return true;
  } catch {
    return false;
  }
}

export function startOfDayInZone(date: Date, timeZone: string): Date {
  return zonedDateToUtc(`${dayKeyInZone(date, timeZone)}T00:00:00`, timeZone);
}

/**
 * Interpret a wall-clock string (`YYYY-MM-DDTHH:mm[:ss]`) in `timeZone` and
 * return the UTC instant. See `zonedWallClockToUtcMs` for the DST policy.
 */
export function zonedDateToUtc(isoLocal: string, timeZone: string): Date {
  const naive = new Date(`${isoLocal}Z`);
  if (Number.isNaN(naive.getTime())) {
    throw new RangeError('Invalid local date-time.');
  }
  return new Date(zonedWallClockToUtcMs(naive.getTime(), timeZone));
}

/**
 * Interpret a local y/m/d h:mi:s tuple (month 1-12) in `timeZone` and return
 * the UTC instant. Out-of-range components are not normalised here; callers
 * that accept untrusted input validate them first.
 */
export function zonedPartsToUtc(
  year: number,
  month: number,
  day: number,
  hour: number,
  minute: number,
  second: number,
  timeZone: string,
): Date {
  return new Date(
    zonedWallClockToUtcMs(Date.UTC(year, month - 1, day, hour, minute, second), timeZone),
  );
}

/**
 * Core wall-clock → UTC conversion. `wallMs` is the wall-clock reading
 * encoded as if it were UTC.
 *
 * The old single-pass version sampled the zone offset at `wallMs` itself —
 * an instant that is hours away from the real one — so for several hours on
 * each side of a DST change it used the wrong offset (America/New_York
 * 2024-03-10 03:30 came out at 08:30Z instead of 07:30Z). Here the offsets in
 * force a day before and a day after are both tried, and an offset is only
 * accepted if converting with it lands on an instant that really has that
 * offset. That confirmation step is what makes the result right on both
 * sides of a transition, in either hemisphere.
 *
 * Policy for the two cases where a wall-clock time does not map to exactly
 * one instant (the same as Temporal's default `disambiguation: 'compatible'`
 * and the JS `Date` constructor):
 *   - Ambiguous (the repeated hour when clocks go back): the earlier instant,
 *     i.e. the first time the clock showed that reading.
 *   - Nonexistent (the skipped hour when clocks go forward): shifted forward
 *     by the gap, i.e. read with the offset in force before the change, so
 *     02:30 on a New York spring-forward day becomes 03:30 EDT.
 */
export function zonedWallClockToUtcMs(wallMs: number, timeZone: string): number {
  if (timeZone === 'UTC' || timeZone === 'Etc/UTC') return wallMs;
  const before = offsetAtMs(wallMs - DAY_MS, timeZone);
  const after = offsetAtMs(wallMs + DAY_MS, timeZone);
  // No transition within a day either side: the offset is unambiguous.
  if (before === after) return wallMs - before;

  const beforeValid = offsetAtMs(wallMs - before, timeZone) === before;
  const afterValid = offsetAtMs(wallMs - after, timeZone) === after;
  if (beforeValid && afterValid) return wallMs - Math.max(before, after); // ambiguous: earlier instant
  if (beforeValid) return wallMs - before;
  if (afterValid) return wallMs - after;
  return wallMs - before; // nonexistent: shift forward across the gap
}

/** Offset of `timeZone` from UTC, in milliseconds, at the given instant. */
export function timeZoneOffsetMs(date: Date, timeZone: string): number {
  return offsetAtMs(date.getTime(), timeZone);
}

function offsetAtMs(ms: number, timeZone: string): number {
  if (timeZone === 'UTC' || timeZone === 'Etc/UTC') return 0;
  const parts = formatterFor('parts', timeZone).formatToParts(new Date(ms));
  let year = 0;
  let month = 1;
  let day = 1;
  let hour = 0;
  let minute = 0;
  let second = 0;
  for (const p of parts) {
    switch (p.type) {
      case 'year':
        year = Number(p.value);
        break;
      case 'month':
        month = Number(p.value);
        break;
      case 'day':
        day = Number(p.value);
        break;
      case 'hour':
        hour = Number(p.value) % 24;
        break;
      case 'minute':
        minute = Number(p.value);
        break;
      case 'second':
        second = Number(p.value);
        break;
    }
  }
  const asUtc = Date.UTC(year, month - 1, day, hour, minute, second);
  // The formatted parts have no milliseconds, so compare against the
  // instant truncated to the second — otherwise every offset is off by the
  // sub-second remainder.
  const wholeSecond = ms - (((ms % 1000) + 1000) % 1000);
  return asUtc - wholeSecond;
}

/** Offset in whole minutes (east-positive) — convenience for import code. */
export function timeZoneOffsetMinutes(date: Date, timeZone: string): number {
  return Math.round(timeZoneOffsetMs(date, timeZone) / MINUTE_MS);
}

/** Hour of day (0-23) in the given zone. */
export function hourInZone(date: Date, timeZone: string): number {
  return Number(formatterFor('hour', timeZone).format(date)) % 24;
}

const WEEKDAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

/** ISO weekday index in the given zone: 0 = Sunday. */
export function weekdayInZone(date: Date, timeZone: string): number {
  return WEEKDAYS.indexOf(formatterFor('weekday', timeZone).format(date));
}

/** YYYY-MM-DD key for grouping records into local days. */
export function dayKeyInZone(date: Date, timeZone: string): string {
  return formatterFor('day', timeZone).format(date);
}

export function daysAgo(n: number, from: Date = new Date()): Date {
  return new Date(from.getTime() - n * DAY_MS);
}

export function minutesBetween(a: Date, b: Date): number {
  return Math.abs(b.getTime() - a.getTime()) / MINUTE_MS;
}

/** "Morning" / "Afternoon" / "Evening" / "Overnight" bucket for an hour. */
export function timeOfDayBucket(hour: number): 'overnight' | 'morning' | 'afternoon' | 'evening' {
  if (hour < 6) return 'overnight';
  if (hour < 12) return 'morning';
  if (hour < 18) return 'afternoon';
  return 'evening';
}

export const TIME_OF_DAY_LABELS = {
  overnight: 'Overnight',
  morning: 'Morning',
  afternoon: 'Afternoon',
  evening: 'Evening',
} as const;

/**
 * Format an instant as the `YYYY-MM-DDTHH:mm` string that a
 * `<input type="datetime-local">` expects, in the given zone — so the default
 * value of a form is the user's wall clock, not the server's.
 */
export function toLocalInputValue(date: Date, timeZone: string): string {
  const parts = Object.fromEntries(
    formatterFor('input', timeZone)
      .formatToParts(date)
      .map((p) => [p.type, p.value]),
  );
  const hour = String(Number(parts.hour) % 24).padStart(2, '0');
  return `${parts.year}-${parts.month}-${parts.day}T${hour}:${parts.minute}`;
}
