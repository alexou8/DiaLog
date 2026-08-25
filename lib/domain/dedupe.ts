import { createHash } from 'node:crypto';

/**
 * Content-addressed key used to make imports idempotent.
 *
 * Two records are the same record when they describe the same measurement at
 * the same instant from the same kind of source. Timestamps are truncated to
 * the minute because meters and exports disagree about seconds, and numeric
 * values are rounded so that float formatting differences between CSV and XML
 * exports of the same reading do not create a duplicate.
 */
export function dedupeKey(parts: {
  type: string;
  takenAt: Date;
  /** Primary numeric value, if the record has one. */
  value?: number | null;
  /** Secondary discriminator, e.g. medication name or activity. */
  discriminator?: string | null;
  /** Stable identifier from the source system, when the export provides one. */
  externalId?: string | null;
}): string {
  const minute = Math.floor(parts.takenAt.getTime() / 60_000);
  const value = parts.value == null ? '' : Math.round(parts.value * 100) / 100;
  const payload = [
    parts.type,
    parts.externalId ?? '',
    minute,
    value,
    (parts.discriminator ?? '').trim().toLowerCase(),
  ].join('|');
  return createHash('sha256').update(payload).digest('hex').slice(0, 32);
}
