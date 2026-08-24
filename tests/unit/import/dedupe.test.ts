import { describe, expect, it } from 'vitest';
import { dedupeRecords, keyRecord } from '@/lib/import/dedupe';
import type { GlucoseRecord, MealRecord, NormalizedRecord } from '@/lib/import/types';

function glucose(overrides: Partial<GlucoseRecord> = {}): GlucoseRecord {
  return {
    kind: 'glucose',
    takenAt: new Date('2026-01-10T08:00:00Z'),
    valueMgdl: 120,
    context: 'UNKNOWN',
    raw: {},
    ...overrides,
  };
}

function meal(overrides: Partial<MealRecord> = {}): MealRecord {
  return {
    kind: 'meal',
    takenAt: new Date('2026-01-10T08:00:00Z'),
    description: 'Oatmeal',
    mealType: 'OTHER',
    raw: {},
    ...overrides,
  };
}

describe('keyRecord', () => {
  it('produces the same key for equivalent glucose records', () => {
    const a = keyRecord(glucose());
    const b = keyRecord(glucose());
    expect(a.dedupeKey).toBe(b.dedupeKey);
  });

  it('produces a different key when the value differs', () => {
    const a = keyRecord(glucose({ valueMgdl: 120 }));
    const b = keyRecord(glucose({ valueMgdl: 130 }));
    expect(a.dedupeKey).not.toBe(b.dedupeKey);
  });

  it('produces a different key when the kind differs', () => {
    const a = keyRecord(glucose());
    const b = keyRecord(meal({ takenAt: glucose().takenAt }));
    expect(a.dedupeKey).not.toBe(b.dedupeKey);
  });

  it('treats timestamps within the same minute as identical', () => {
    const a = keyRecord(glucose({ takenAt: new Date('2026-01-10T08:00:00Z') }));
    const b = keyRecord(glucose({ takenAt: new Date('2026-01-10T08:00:45Z') }));
    expect(a.dedupeKey).toBe(b.dedupeKey);
  });

  it('treats timestamps in different minutes as different', () => {
    const a = keyRecord(glucose({ takenAt: new Date('2026-01-10T08:00:00Z') }));
    const b = keyRecord(glucose({ takenAt: new Date('2026-01-10T08:01:00Z') }));
    expect(a.dedupeKey).not.toBe(b.dedupeKey);
  });
});

describe('dedupeRecords', () => {
  it('treats all distinct records as fresh when there are no existing keys', () => {
    const records: NormalizedRecord[] = [glucose({ valueMgdl: 100 }), glucose({ valueMgdl: 110 })];
    const { fresh, duplicates } = dedupeRecords(records);
    expect(fresh).toHaveLength(2);
    expect(duplicates).toHaveLength(0);
  });

  it('detects duplicates within the same file (first occurrence wins)', () => {
    const records: NormalizedRecord[] = [glucose(), glucose()];
    const { fresh, duplicates } = dedupeRecords(records);
    expect(fresh).toHaveLength(1);
    expect(duplicates).toHaveLength(1);
  });

  it('detects duplicates against a supplied set of existing keys', () => {
    const record = glucose();
    const existingKey = keyRecord(record).dedupeKey;
    const { fresh, duplicates } = dedupeRecords([record], new Set([existingKey]));
    expect(fresh).toHaveLength(0);
    expect(duplicates).toHaveLength(1);
  });

  it('is a pure function: does not mutate its inputs', () => {
    const records: NormalizedRecord[] = [glucose()];
    const existing = new Set(['abc']);
    dedupeRecords(records, existing);
    expect(existing.size).toBe(1);
    expect(records).toHaveLength(1);
  });
});
