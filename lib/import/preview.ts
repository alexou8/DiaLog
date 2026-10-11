/**
 * User-facing text for the import preview rows.
 *
 * Kept out of the Server Action so it can be unit-tested, and so glucose is
 * shown the way every other screen shows it: converted to the user's
 * display unit through lib/domain/units.ts. The preview used to print
 * "99 mg/dL (stored value)" to everyone, which showed mmol/L users a number
 * in a unit they never chose — the Units invariant says display is always
 * the user's preference.
 */
import type { GlucoseUnit } from '@prisma/client';
import { formatGlucoseWithUnit } from '@/lib/domain/units';
import type { NormalizedRecord } from './types';

export function previewDetail(record: NormalizedRecord, unit: GlucoseUnit, locale: string): string {
  switch (record.kind) {
    case 'glucose':
      return formatGlucoseWithUnit(record.valueMgdl, unit, locale);
    case 'meal':
      return record.description;
    case 'medication':
      return record.name;
    case 'exercise':
      return `${record.activity}, ${record.durationMin} min`;
    default:
      return 'Not recorded';
  }
}
