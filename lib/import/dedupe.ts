/**
 * Import-time duplicate detection.
 *
 * Pure function: the caller (which has database access) supplies the set of
 * dedupeKeys already stored for the user, and this module tells them which
 * incoming records are fresh vs. duplicates — both against that existing set
 * and against each other within the same file (re-uploading the same export
 * twice, or a file that overlaps a previous one, must not double-import).
 */
import { dedupeKey } from '@/lib/domain/dedupe';
import type { NormalizedRecord } from './types';

export interface KeyedRecord<T extends NormalizedRecord = NormalizedRecord> {
  record: T;
  dedupeKey: string;
}

export interface DedupeResult {
  fresh: KeyedRecord[];
  /** Duplicates, each paired with the key that already "claimed" this dedupeKey. */
  duplicates: KeyedRecord[];
}

function valueFor(record: NormalizedRecord): number | null {
  switch (record.kind) {
    case 'glucose':
      return record.valueMgdl;
    case 'weight':
      return record.weightKg;
    case 'exercise':
      return record.durationMin;
    case 'sleep':
      return record.durationMin;
    default:
      return null;
  }
}

function discriminatorFor(record: NormalizedRecord): string | null {
  switch (record.kind) {
    case 'glucose':
      return record.context;
    case 'meal':
      return record.description;
    case 'medication':
      return record.name;
    case 'exercise':
      return record.activity;
    case 'note':
      return record.text;
    default:
      return null;
  }
}

function externalIdFor(record: NormalizedRecord): string | null {
  return 'externalId' in record ? (record.externalId ?? null) : null;
}

/** Computes the stable dedupeKey for a single normalized record. */
export function keyRecord(record: NormalizedRecord): KeyedRecord {
  const key = dedupeKey({
    type: record.kind,
    takenAt: record.takenAt,
    value: valueFor(record),
    discriminator: discriminatorFor(record),
    externalId: externalIdFor(record),
  });
  return { record, dedupeKey: key };
}

/**
 * Splits `records` into `fresh` and `duplicates`, checking both against
 * `existingKeys` (already persisted for this user) and against earlier
 * records in the same batch (first occurrence wins, later ones are
 * duplicates of it).
 */
export function dedupeRecords(
  records: readonly NormalizedRecord[],
  existingKeys: ReadonlySet<string> | Iterable<string> = [],
): DedupeResult {
  const seen = new Set<string>(existingKeys);
  const fresh: KeyedRecord[] = [];
  const duplicates: KeyedRecord[] = [];

  for (const record of records) {
    const keyed = keyRecord(record);
    if (seen.has(keyed.dedupeKey)) {
      duplicates.push(keyed);
    } else {
      seen.add(keyed.dedupeKey);
      fresh.push(keyed);
    }
  }

  return { fresh, duplicates };
}
