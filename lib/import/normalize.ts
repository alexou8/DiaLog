/**
 * Value normalisation shared by connectors: glucose unit inference/conversion
 * and plausibility checks (via lib/domain/units.ts), free-text context
 * mapping to the GlucoseContext enum, and the storage-constraint checks that
 * keep unstorable numbers from reaching Prisma.
 */
import { inferGlucoseUnit, isPlausibleGlucose, toMgdl } from '@/lib/domain/units';
import type { GlucoseContext, GlucoseUnit } from '@prisma/client';
import type { NormalizedRecord, ParseResult, RowIssue, RowIssueCode } from './types';

export { inferGlucoseUnit };

export interface GlucoseValueResult {
  valueMgdl: number | null;
  /** Set when the value could not be trusted. */
  issue?: 'INVALID_VALUE' | 'OUT_OF_RANGE' | 'UNKNOWN_UNIT';
  message?: string;
}

/**
 * Converts a raw numeric glucose value into canonical mg/dL, given a known or
 * inferred unit, rejecting implausible values (typos, unit mix-ups) rather
 * than silently storing them.
 */
export function normalizeGlucoseValue(
  raw: number | null | undefined,
  unit: GlucoseUnit | null,
): GlucoseValueResult {
  if (raw === null || raw === undefined || !Number.isFinite(raw)) {
    return {
      valueMgdl: null,
      issue: 'INVALID_VALUE',
      message: 'Glucose value is not a valid number.',
    };
  }
  if (!unit) {
    return {
      valueMgdl: null,
      issue: 'UNKNOWN_UNIT',
      message: `Could not determine whether ${raw} is mg/dL or mmol/L.`,
    };
  }
  if (!isPlausibleGlucose(raw, unit)) {
    return {
      valueMgdl: null,
      issue: 'OUT_OF_RANGE',
      message: `Value ${raw} ${unit === 'MMOLL' ? 'mmol/L' : 'mg/dL'} is outside the plausible range for a glucose reading.`,
    };
  }
  return { valueMgdl: toMgdl(raw, unit) };
}

/**
 * Maps free-text context descriptions (as seen in device exports and manual
 * logs) to the GlucoseContext enum. Case/space/punctuation-insensitive.
 */
export function mapGlucoseContext(text: string | null | undefined): GlucoseContext {
  if (!text) return 'UNKNOWN';
  const t = text.toLowerCase().trim();
  if (t.length === 0) return 'UNKNOWN';

  if (/\bfasting\b|\bfast(ed)?\b|before\s*wake|wake\s*up|am\s*fasting/.test(t)) return 'FASTING';
  if (
    /before\s*(breakfast|lunch|dinner|meal|eating)|pre[\s-]?(meal|breakfast|lunch|dinner|prandial)/.test(
      t,
    )
  ) {
    return 'BEFORE_MEAL';
  }
  if (
    /after\s*(breakfast|lunch|dinner|meal|eating)|post[\s-]?(meal|breakfast|lunch|dinner|prandial)/.test(
      t,
    )
  ) {
    return 'AFTER_MEAL';
  }
  if (/bed\s*time|bedtime|before\s*sleep|before\s*bed|night\s*time reading|hs\b/.test(t))
    return 'BEDTIME';
  if (/random|any\s*time|casual|spot\s*check/.test(t)) return 'RANDOM';
  return 'UNKNOWN';
}

/**
 * Convenience wrapper that infers the unit for a whole file (once) when the
 * caller has not been told the unit explicitly, per the heuristic in
 * lib/domain/units.ts: it only trusts values that look unambiguous.
 */
export function detectFileGlucoseUnit(
  values: readonly number[],
  explicit?: GlucoseUnit | null,
): GlucoseUnit | null {
  if (explicit) return explicit;
  return inferGlucoseUnit(values);
}

// ------------------------------------------------- storage constraints

/** Postgres `integer` (int4) range — what Prisma `Int` columns map to. */
export const INT4_MIN = -2_147_483_648;
export const INT4_MAX = 2_147_483_647;

/**
 * The `Int` columns each record kind writes (see prisma/schema.prisma). Every
 * one of them is a count or a duration, so negatives are rejected too.
 * tests/unit/import/storage-constraints.test.ts reads the schema and fails if
 * an importable model gains an Int or Float column missing from these lists.
 * (Medication `dose` is a String column, and blood pressure, hydration,
 * symptom and mood are not importable kinds, so none of them belong here.)
 */
export const INT_FIELDS = {
  exercise: ['durationMin', 'steps'],
  sleep: ['durationMin', 'quality'],
} as const satisfies Partial<Record<NormalizedRecord['kind'], readonly string[]>>;

/** Numeric `Float` columns, which only need to be finite. */
export const FLOAT_FIELDS = {
  glucose: ['valueMgdl'],
  meal: ['carbsG', 'proteinG', 'fatG', 'fiberG', 'calories'],
  exercise: ['distanceKm'],
  weight: ['weightKg'],
} as const satisfies Partial<Record<NormalizedRecord['kind'], readonly string[]>>;

export interface StorageIssue {
  code: Extract<RowIssueCode, 'INVALID_VALUE' | 'OUT_OF_RANGE'>;
  message: string;
}

/** Checks one value destined for an `Int` column. Null/undefined (optional column) passes. */
export function checkStorableInt(value: unknown, field: string): StorageIssue | null {
  if (value === null || value === undefined) return null;
  if (typeof value !== 'number' || !Number.isFinite(value) || !Number.isInteger(value)) {
    return { code: 'INVALID_VALUE', message: `${field} must be a whole number.` };
  }
  if (value < 0 || value > INT4_MAX) {
    return { code: 'OUT_OF_RANGE', message: `${field} is outside the range DiaLog can store.` };
  }
  return null;
}

/**
 * Why this exists: generic XML/JSON pass user-supplied numbers straight into
 * `Int` columns. `durationMin="12.5"` or `steps="3000000000"` survived the
 * preview, then Prisma (Float for Int) or Postgres (int4 overflow) rejected
 * the whole commit transaction at write time — one bad row failed every row.
 * A value that cannot be stored is a row problem, so it becomes a RowIssue
 * before anything reaches the database. Messages name the field, never the
 * value.
 */
export function storageIssueFor(record: NormalizedRecord): StorageIssue | null {
  const fields = record as unknown as Record<string, unknown>;
  for (const field of (INT_FIELDS as Partial<Record<string, readonly string[]>>)[record.kind] ??
    []) {
    const issue = checkStorableInt(fields[field], field);
    if (issue) return issue;
  }
  for (const field of (FLOAT_FIELDS as Partial<Record<string, readonly string[]>>)[record.kind] ??
    []) {
    const value = fields[field];
    if (value === null || value === undefined) continue;
    if (typeof value !== 'number' || !Number.isFinite(value)) {
      return { code: 'INVALID_VALUE', message: `${field} is not a valid number.` };
    }
  }
  return null;
}

/**
 * Per-row form for connectors: pushes the record when it is storable,
 * otherwise records a RowIssue with the real row number.
 */
export function pushIfStorable(
  records: NormalizedRecord[],
  issues: RowIssue[],
  record: NormalizedRecord,
  rowNumber: number,
  rawRow?: string,
): void {
  const issue = storageIssueFor(record);
  if (issue) issues.push({ rowNumber, ...issue, ...(rawRow === undefined ? {} : { rawRow }) });
  else records.push(record);
}

/**
 * Backstop applied to every connector's output by the import service, so a
 * connector that forgets the per-row check still cannot send an unstorable
 * value to Prisma. Records carry no row number, so these issues use row 0
 * ("row unknown") — connectors should prefer `pushIfStorable`.
 */
export function enforceStorageConstraints(result: ParseResult): ParseResult {
  const records: NormalizedRecord[] = [];
  const extraIssues: RowIssue[] = [];
  for (const record of result.records) {
    const issue = storageIssueFor(record);
    if (issue) {
      extraIssues.push({
        rowNumber: 0,
        code: issue.code,
        message: `A ${record.kind} record could not be stored: ${issue.message}`,
      });
    } else {
      records.push(record);
    }
  }
  if (extraIssues.length === 0) return result;
  return { ...result, records, issues: [...result.issues, ...extraIssues] };
}
