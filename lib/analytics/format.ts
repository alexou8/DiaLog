/**
 * Display formatting for analytics statements.
 *
 * Findings are computed in the canonical storage unit (mg/dL) but are read by
 * a person, so the wording has to be in the unit that person actually uses.
 * Everything that turns a number into words goes through here.
 */
import type { GlucoseUnit } from '@prisma/client';
import { fromMgdl, unitLabel } from '@/lib/domain/units';

export type DisplayUnit = GlucoseUnit;

/** A glucose level, e.g. "8.2 mmol/L" or "148 mg/dL". */
export function formatLevel(mgdl: number, unit: DisplayUnit = 'MGDL'): string {
  const value = fromMgdl(mgdl, unit);
  return `${value.toFixed(unit === 'MMOLL' ? 1 : 0)} ${unitLabel(unit)}`;
}

/**
 * A difference between two levels. Conversion is linear with no offset, so a
 * delta converts exactly like a level does.
 */
export function formatDelta(mgdlDelta: number, unit: DisplayUnit = 'MGDL'): string {
  return formatLevel(Math.abs(mgdlDelta), unit);
}
