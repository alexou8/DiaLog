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

/** Format a canonical mg/dL value for display, without the unit suffix. */
export function formatGlucose(mgdl: number, unit: GlucoseUnit, locale = 'en-CA'): string {
  const digits = unitPrecision(unit);
  return new Intl.NumberFormat(locale, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(fromMgdl(mgdl, unit));
}

/** Format a canonical mg/dL value with its unit suffix, e.g. "7.2 mmol/L". */
export function formatGlucoseWithUnit(mgdl: number, unit: GlucoseUnit, locale = 'en-CA'): string {
  return `${formatGlucose(mgdl, unit, locale)} ${unitLabel(unit)}`;
}

/**
 * Plausible entry bounds per unit. Values outside these are almost certainly
 * a unit mix-up or a typo, so entry and import both reject them rather than
 * silently storing an implausible reading.
 */
export const GLUCOSE_ENTRY_BOUNDS: Record<GlucoseUnit, { min: number; max: number }> = {
  MGDL: { min: 20, max: 700 },
  MMOLL: { min: 1.1, max: 38.9 },
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
  const finite = values.filter((v) => Number.isFinite(v) && v > 0);
  if (finite.length === 0) return null;
  const max = Math.max(...finite);
  if (max > 40) return 'MGDL';
  const median = [...finite].sort((a, b) => a - b)[Math.floor(finite.length / 2)] ?? 0;
  if (max <= 40 && median < 30) return 'MMOLL';
  return null;
}

export const KG_PER_LB = 0.45359237;
export const ML_PER_FL_OZ = 29.5735;
export const KM_PER_MILE = 1.609344;
