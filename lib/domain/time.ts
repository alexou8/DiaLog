/** Locale- and timezone-aware date helpers. All storage is UTC. */

export function startOfDayInZone(date: Date, timeZone: string): Date {
  const parts = new Intl.DateTimeFormat('en-CA', {
    timeZone,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  }).format(date);
  return zonedDateToUtc(`${parts}T00:00:00`, timeZone);
}

/** Interpret a wall-clock string in `timeZone` and return the UTC instant. */
export function zonedDateToUtc(isoLocal: string, timeZone: string): Date {
  const naive = new Date(`${isoLocal}Z`);
  const offset = timeZoneOffsetMs(naive, timeZone);
  return new Date(naive.getTime() - offset);
}

/** Offset of `timeZone` from UTC, in milliseconds, at the given instant. */
export function timeZoneOffsetMs(date: Date, timeZone: string): number {
  const dtf = new Intl.DateTimeFormat('en-US', {
    timeZone,
    hour12: false,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
  const parts = Object.fromEntries(dtf.formatToParts(date).map((p) => [p.type, p.value]));
  const asUtc = Date.UTC(
    Number(parts.year),
    Number(parts.month) - 1,
    Number(parts.day),
    Number(parts.hour) % 24,
    Number(parts.minute),
    Number(parts.second),
  );
  return asUtc - date.getTime();
}

/** Hour of day (0-23) in the given zone. */
export function hourInZone(date: Date, timeZone: string): number {
  return (
    Number(
      new Intl.DateTimeFormat('en-US', { timeZone, hour: '2-digit', hour12: false }).format(date),
    ) % 24
  );
}

/** ISO weekday index in the given zone: 0 = Sunday. */
export function weekdayInZone(date: Date, timeZone: string): number {
  const name = new Intl.DateTimeFormat('en-US', { timeZone, weekday: 'short' }).format(date);
  return ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'].indexOf(name);
}

/** YYYY-MM-DD key for grouping records into local days. */
export function dayKeyInZone(date: Date, timeZone: string): string {
  return new Intl.DateTimeFormat('en-CA', {
    timeZone,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  }).format(date);
}

export function daysAgo(n: number, from: Date = new Date()): Date {
  return new Date(from.getTime() - n * 86_400_000);
}

export function minutesBetween(a: Date, b: Date): number {
  return Math.abs(b.getTime() - a.getTime()) / 60_000;
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
    new Intl.DateTimeFormat('en-CA', {
      timeZone,
      hour12: false,
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    })
      .formatToParts(date)
      .map((p) => [p.type, p.value]),
  );
  const hour = String(Number(parts.hour) % 24).padStart(2, '0');
  return `${parts.year}-${parts.month}-${parts.day}T${hour}:${parts.minute}`;
}
