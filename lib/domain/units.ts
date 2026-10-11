/**
 * Unit conversion and formatting.
 *
 * Glucose is stored canonically in mg/dL. Everything the user sees goes
 * through here so that switching units never means touching UI code.
 */
import type { GlucoseUnit } from '@prisma/client';

/** Exact molar mass conversion factor for glucose (g/mol basis). */
export const MGDL_PER_MMOLL = 18.0182;

export function mmollToMgdl(mmoll: number): number {
  return mmoll * MGDL_PER_MMOLL;
}

export function mgdlToMmoll(mgdl: number): number {
  return mgdl / MGDL_PER_MMOLL;
}

/** Convert a canonical mg/dL value into the user's preferred unit. */
export function fromMgdl(mgdl: number, unit: GlucoseUnit): number {
  return unit === 'MMOLL' ? mgdlToMmoll(mgdl) : mgdl;
}

/** Convert a user-entered value in `unit` into canonical mg/dL. */
export function toMgdl(value: number, unit: GlucoseUnit): number {
  return unit === 'MMOLL' ? mmollToMgdl(value) : value;
}

export function unitLabel(unit: GlucoseUnit): string {
  return unit === 'MMOLL' ? 'mmol/L' : 'mg/dL';
}

/** Decimal places conventionally used for each unit. */
export function unitPrecision(unit: GlucoseUnit): number {
  return unit === 'MMOLL' ? 1 : 0;
}

/**
 * Charts call `formatGlucose` about twice per plotted point, and building an
 * `Intl.NumberFormat` costs tens of microseconds, so instances are cached per
 * (locale, unit). The cap only stops unbounded growth from odd locale strings.
 */
const NUMBER_FORMAT_CACHE_LIMIT = 256;
const numberFormatCache = new Map<string, Intl.NumberFormat>();

function glucoseNumberFormat(locale: string, unit: GlucoseUnit): Intl.NumberFormat {
  const key = `${locale}|${unit}`;
  let nf = numberFormatCache.get(key);
  if (!nf) {
    const digits = unitPrecision(unit);
    nf = new Intl.NumberFormat(locale, {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    });
    if (numberFormatCache.size >= NUMBER_FORMAT_CACHE_LIMIT) numberFormatCache.clear();
    numberFormatCache.set(key, nf);
  }
  return nf;
}

/** Format a canonical mg/dL value for display, without the unit suffix. */
export function formatGlucose(mgdl: number, unit: GlucoseUnit, locale = 'en-CA'): string {
  return glucoseNumberFormat(locale, unit).format(fromMgdl(mgdl, unit));
}

/** Format a canonical mg/dL value with its unit suffix, e.g. "7.2 mmol/L". */
export function formatGlucoseWithUnit(mgdl: number, unit: GlucoseUnit, locale = 'en-CA'): string {
  return `${formatGlucose(mgdl, unit, locale)} ${unitLabel(unit)}`;
}

/** Round to a unit's conventional display precision. */
export function roundToUnitPrecision(value: number, unit: GlucoseUnit): number {
  const factor = 10 ** unitPrecision(unit);
  return Math.round(value * factor) / factor;
}

/** Canonical plausibility bounds, in mg/dL. */
const CANONICAL_ENTRY_BOUNDS_MGDL = { min: 20, max: 700 } as const;

/**
 * Plausible entry bounds per unit. Values outside these are almost certainly
 * a unit mix-up or a typo, so entry and import both reject them rather than
 * silently storing an implausible reading.
 *
 * The mmol/L bounds are derived from the canonical mg/dL ones at mmol/L
 * display precision rather than typed in separately, so the two can never
 * drift apart: each mmol/L bound is exactly what the mg/dL bound displays as
 * (20 mg/dL shows as 1.1, 700 as 38.9). One decimal of mmol/L is ~1.8 mg/dL,
 * so the converted bounds land within a display step of 20/700 (19.8/700.9)
 * — the same reading a meter would show, not a looser rule. Keeping 1.1 also
 * matters: it is the "LO" floor many meters report, a real severe low.
 */
export const GLUCOSE_ENTRY_BOUNDS: Record<GlucoseUnit, { min: number; max: number }> = {
  MGDL: { ...CANONICAL_ENTRY_BOUNDS_MGDL },
  MMOLL: {
    min: roundToUnitPrecision(mgdlToMmoll(CANONICAL_ENTRY_BOUNDS_MGDL.min), 'MMOLL'),
    max: roundToUnitPrecision(mgdlToMmoll(CANONICAL_ENTRY_BOUNDS_MGDL.max), 'MMOLL'),
  },
};

export function isPlausibleGlucose(value: number, unit: GlucoseUnit): boolean {
  const b = GLUCOSE_ENTRY_BOUNDS[unit];
  return Number.isFinite(value) && value >= b.min && value <= b.max;
}

/**
 * Heuristic used by importers when a file does not declare its unit: real
 * mmol/L readings essentially never exceed 40, and mg/dL readings essentially
 * never fall below 20.
 */
export function inferGlucoseUnit(values: readonly number[]): GlucoseUnit | null {
  // A loop, not `Math.max(...values)`: spreading passes every element as a
  // call argument, and V8 throws RangeError past ~120k of them — which a
  // single CGM export easily exceeds.
  const finite: number[] = [];
  let max = -Infinity;
  for (const v of values) {
    if (!Number.isFinite(v) || v <= 0) continue;
    finite.push(v);
    if (v > max) max = v;
  }
  if (finite.length === 0) return null;
  if (max > 40) return 'MGDL';
  finite.sort((a, b) => a - b);
  const median = finite[Math.floor(finite.length / 2)] ?? 0;
  if (median < 30) return 'MMOLL';
  return null;
}

/** Largest finite value, or null. Loop-based so it is safe on any array size. */
export function maxOf(values: readonly number[]): number | null {
  let max: number | null = null;
  for (const v of values) {
    if (Number.isFinite(v) && (max === null || v > max)) max = v;
  }
  return max;
}

/** Smallest finite value, or null. Loop-based so it is safe on any array size. */
export function minOf(values: readonly number[]): number | null {
  let min: number | null = null;
  for (const v of values) {
    if (Number.isFinite(v) && (min === null || v < min)) min = v;
  }
  return min;
}

/**
 * Convert a user-entered target-range bound to canonical mg/dL, snapped to a
 * whole mg/dL. Readings are stored at whole-mg/dL resolution in practice
 * (meters and CGMs report integers), so an unrounded target such as 3.9 mmol/L
 * → 70.27 mg/dL made a reading of 70 — which displays as 3.9, equal to the
 * target — count as below range, understating time-in-range for mmol users.
 */
export function targetToMgdl(value: number, unit: GlucoseUnit): number {
  return Math.round(toMgdl(value, unit));
}

export const KG_PER_LB = 0.45359237;
export const ML_PER_FL_OZ = 29.5735;
export const KM_PER_MILE = 1.609344;
