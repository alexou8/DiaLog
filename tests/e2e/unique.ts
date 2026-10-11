/**
 * Per-attempt unique values for tests that create records.
 *
 * DiaLog gives every record a content-addressed dedupe key built from its
 * type, its timestamp truncated to the minute, and its value (see
 * lib/domain/dedupe.ts). That is deliberate — it makes re-importing a file a
 * no-op — but it means a test that saves fixed content at "now" cannot run
 * twice inside the same minute: the second save is correctly rejected as a
 * duplicate. Two specs writing similar content to the shared account, or a
 * re-run against an account that was not reset, do exactly that.
 *
 * So every test that creates a record must vary its content. These helpers are
 * the single place that happens, rather than each spec inventing its own.
 */
let sequence = 0;

/**
 * Seeded randomness, so a failing run can be reproduced.
 *
 * Every value below used to come from `Math.random()` and `Date.now()`, so a
 * failure that depended on the data (a dedupe collision, a value that lands in
 * a different status band) could not be replayed: the run that failed left no
 * record of what it had generated. Values now derive from one seed.
 * `announceSeed()` (called once by global-setup) prints it and publishes it
 * through `PW_SEED`, which is how it reaches the worker processes; set `PW_SEED`
 * yourself to replay a run.
 *
 * Each worker mixes its own `TEST_WORKER_INDEX` into the stream. Without that,
 * every worker would replay the identical sequence from the start, and a worker
 * Playwright restarts after a failure would regenerate the values the failed
 * one already saved — colliding with its own records on the dedupe key.
 *
 * The seed makes the *sequence* reproducible; uniqueness within a run still
 * comes from `sequence` and from the account being reset before every run
 * (global-setup.ts), not from the clock.
 */
export function announceSeed(): string {
  if (!process.env.PW_SEED) {
    process.env.PW_SEED = Math.floor(Math.random() * 0xffffffff)
      .toString(16)
      .padStart(8, '0');
    console.log(
      `[e2e] PW_SEED=${process.env.PW_SEED} — set PW_SEED to this value to reproduce the generated test data.`,
    );
  } else {
    console.log(`[e2e] PW_SEED=${process.env.PW_SEED} (from the environment)`);
  }
  return process.env.PW_SEED;
}

/** FNV-1a, to turn the seed string into the 32-bit state mulberry32 needs. */
function hashSeed(text: string): number {
  let hash = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
}

function mulberry32(seed: number): () => number {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

let stream: (() => number) | undefined;

/** Seeded replacement for `Math.random()` in e2e data generation. */
export function random(): number {
  // Resolved lazily: global-setup publishes PW_SEED before any worker starts,
  // but this module can be imported earlier than that in the main process.
  stream ??= mulberry32(
    hashSeed(`${process.env.PW_SEED ?? 'unseeded'}:${process.env.TEST_WORKER_INDEX ?? 'main'}`),
  );
  return stream();
}

/** Short token that differs on every call within a process, and between workers. */
export function uniqueToken(): string {
  sequence += 1;
  return `${sequence.toString(36)}${Math.floor(random() * 1679616)
    .toString(36)
    .padStart(4, '0')}`;
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
    ? (random() * 9 + 4.5).toFixed(1) // 4.5-13.5 mmol/L
    : String(Math.floor(random() * 170) + 85); // 85-255 mg/dL
}

/** A duration in minutes that differs per attempt but stays realistic. */
export function uniqueDurationMinutes(min = 15, max = 90): string {
  return String(Math.floor(random() * (max - min)) + min);
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
  return Math.floor(random() * 20000) + 1;
}
