/**
 * Per-attempt unique values for tests that create records.
 *
 * DiaLog gives every record a content-addressed dedupe key built from its
 * type, its timestamp truncated to the minute, and its value (see
 * lib/domain/dedupe.ts). That is deliberate — it makes re-importing a file a
 * no-op — but it means a test that saves fixed content at "now" cannot run
 * twice inside the same minute: the second save is correctly rejected as a
 * duplicate. Playwright retries do exactly that.
 *
 * So every test that creates a record must vary its content. These helpers are
 * the single place that happens, rather than each spec inventing its own.
 */
let sequence = 0;

/** Short token that differs on every call within a process, and between runs. */
export function uniqueToken(): string {
  sequence += 1;
  return `${Date.now().toString(36)}${sequence}${Math.floor(Math.random() * 1296).toString(36)}`;
}

/** Free-text value that will not collide with another attempt's. */
export function uniqueText(prefix: string): string {
  return `${prefix} ${uniqueToken()}`;
}

/**
 * A plausible glucose reading, in whichever unit the form is showing.
 * `labelText` is the field's own label, e.g. "Your reading (mg/dL)".
 */
export function uniqueGlucoseValue(labelText: string): string {
  return labelText.includes('mmol/L')
    ? (Math.random() * 9 + 4.5).toFixed(1) // 4.5-13.5 mmol/L
    : String(Math.floor(Math.random() * 170) + 85); // 85-255 mg/dL
}

/** A duration in minutes that differs per attempt but stays realistic. */
export function uniqueDurationMinutes(min = 15, max = 90): string {
  return String(Math.floor(Math.random() * (max - min)) + min);
}

/**
 * Shift a `datetime-local` value backwards by a unique number of minutes, so
 * the resulting record lands in its own minute bucket.
 */
export function shiftLocalDateTime(value: string, minutesEarlier: number): string {
  const [datePart = '', timePart = '00:00'] = value.split('T');
  const [year = '1970', month = '01', day = '01'] = datePart.split('-');
  const [hour = '00', minute = '00'] = timePart.split(':');
  const shifted = new Date(
    Date.UTC(Number(year), Number(month) - 1, Number(day), Number(hour), Number(minute)),
  );
  shifted.setUTCMinutes(shifted.getUTCMinutes() - minutesEarlier);
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${shifted.getUTCFullYear()}-${pad(shifted.getUTCMonth() + 1)}-${pad(shifted.getUTCDate())}T${pad(shifted.getUTCHours())}:${pad(shifted.getUTCMinutes())}`;
}

/** A unique number of minutes to shift a record into its own bucket. */
export function uniqueMinuteOffset(): number {
  return Math.floor(Math.random() * 20000) + 1;
}
