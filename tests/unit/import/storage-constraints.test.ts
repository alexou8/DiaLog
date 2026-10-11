/**
 * A number Prisma cannot write to an `Int` column (fractional, or outside
 * Postgres int4) used to pass the preview and then fail the whole commit
 * transaction. It must become a RowIssue at parse time instead.
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { genericJsonConnector } from '@/lib/import/connectors/generic-json';
import { genericXmlConnector } from '@/lib/import/connectors/generic-xml';
import {
  FLOAT_FIELDS,
  INT4_MAX,
  INT_FIELDS,
  checkStorableInt,
  enforceStorageConstraints,
  storageIssueFor,
} from '@/lib/import/normalize';
import { parseXmlText } from '@/lib/import/parse';
import type { NormalizedRecord, ParseResult } from '@/lib/import/types';

const NOW = new Date('2026-08-24T12:00:00Z');
const T = new Date('2026-01-10T08:00:00Z');

describe('checkStorableInt', () => {
  it('accepts whole numbers in range and absent optional values', () => {
    expect(checkStorableInt(30, 'durationMin')).toBeNull();
    expect(checkStorableInt(INT4_MAX, 'steps')).toBeNull();
    expect(checkStorableInt(null, 'steps')).toBeNull();
    expect(checkStorableInt(undefined, 'steps')).toBeNull();
  });

  it('rejects fractional, non-finite, negative and over-int4 values', () => {
    expect(checkStorableInt(12.5, 'durationMin')?.code).toBe('INVALID_VALUE');
    expect(checkStorableInt(Number.NaN, 'durationMin')?.code).toBe('INVALID_VALUE');
    expect(checkStorableInt(1e12, 'steps')?.code).toBe('OUT_OF_RANGE');
    expect(checkStorableInt(INT4_MAX + 1, 'steps')?.code).toBe('OUT_OF_RANGE');
    expect(checkStorableInt(-5, 'durationMin')?.code).toBe('OUT_OF_RANGE');
  });

  it('never puts the value itself in the message', () => {
    expect(checkStorableInt(1e12, 'steps')?.message).not.toMatch(/1e\+?12|1000000000000/);
  });
});

describe('storageIssueFor', () => {
  it('checks the Int columns of exercise and sleep records', () => {
    const exercise: NormalizedRecord = {
      kind: 'exercise',
      takenAt: T,
      activity: 'walk',
      durationMin: 30,
      steps: 3_000_000_000,
      raw: {},
    };
    expect(storageIssueFor(exercise)?.code).toBe('OUT_OF_RANGE');
    const sleep: NormalizedRecord = {
      kind: 'sleep',
      takenAt: T,
      endedAt: T,
      durationMin: 420,
      quality: 2.5,
      raw: {},
    };
    expect(storageIssueFor(sleep)?.code).toBe('INVALID_VALUE');
  });

  it('passes ordinary records of every kind', () => {
    expect(
      storageIssueFor({
        kind: 'glucose',
        takenAt: T,
        valueMgdl: 99.1,
        context: 'UNKNOWN',
        raw: {},
      }),
    ).toBeNull();
    expect(storageIssueFor({ kind: 'note', takenAt: T, text: 'x', raw: {} })).toBeNull();
  });
});

describe('enforceStorageConstraints (service backstop)', () => {
  it('moves unstorable records into issues and keeps the rest', () => {
    const result: ParseResult = {
      records: [
        { kind: 'exercise', takenAt: T, activity: 'run', durationMin: 12.5, raw: {} },
        { kind: 'exercise', takenAt: T, activity: 'run', durationMin: 20, raw: {} },
      ],
      issues: [],
      rowsTotal: 2,
      detectedUnit: null,
      warnings: [],
    };
    const checked = enforceStorageConstraints(result);
    expect(checked.records).toHaveLength(1);
    expect(checked.issues).toHaveLength(1);
    expect(checked.issues[0]?.code).toBe('INVALID_VALUE');
  });

  it('returns the same result object when nothing needs moving', () => {
    const result: ParseResult = {
      records: [],
      issues: [],
      rowsTotal: 0,
      detectedUnit: null,
      warnings: [],
    };
    expect(enforceStorageConstraints(result)).toBe(result);
  });
});

describe('generic connectors report unstorable numbers per row', () => {
  it('generic-xml: fractional durationMin and 1e12 steps become RowIssues', async () => {
    const xml = parseXmlText(
      [
        '<records>',
        '<record type="exercise" takenAt="2026-01-10T08:00:00Z" activity="walk" durationMin="12.5"/>',
        '<record type="exercise" takenAt="2026-01-10T09:00:00Z" activity="walk" durationMin="20" steps="1e12"/>',
        '<record type="exercise" takenAt="2026-01-10T10:00:00Z" activity="walk" durationMin="20" steps="2400"/>',
        '</records>',
      ].join(''),
    );
    const result = await genericXmlConnector.parse(
      { filename: 'x.xml', mimeType: 'application/xml', xml },
      { now: NOW },
    );
    expect(result.records).toHaveLength(1);
    expect(result.issues.map((i) => [i.rowNumber, i.code])).toEqual([
      [1, 'INVALID_VALUE'],
      [2, 'OUT_OF_RANGE'],
    ]);
  });

  it('generic-json: fractional sleep quality becomes a RowIssue', async () => {
    const result = await genericJsonConnector.parse(
      {
        filename: 'x.json',
        mimeType: 'application/json',
        json: [
          {
            type: 'sleep',
            takenAt: '2026-01-09T23:00:00Z',
            endedAt: '2026-01-10T07:00:00Z',
            durationMin: 480,
            quality: 3.7,
          },
        ],
      },
      { now: NOW },
    );
    expect(result.records).toHaveLength(0);
    expect(result.issues[0]?.code).toBe('INVALID_VALUE');
  });
});

describe('storage backstop coverage', () => {
  // The backstop only protects the columns it lists. Pin the lists to the
  // schema so a numeric column added to an importable model cannot be left
  // unchecked (and fail a whole commit) without this test noticing.
  const MODEL_BY_KIND: Record<NormalizedRecord['kind'], string> = {
    glucose: 'GlucoseReading',
    meal: 'Meal',
    medication: 'MedicationEvent',
    exercise: 'ExerciseSession',
    sleep: 'SleepSession',
    weight: 'WeightMeasurement',
    note: 'NoteEntry',
  };
  // Line endings normalised: a Windows checkout may have CRLF.
  const schema = readFileSync(join(process.cwd(), 'prisma', 'schema.prisma'), 'utf8').replace(
    /\r\n/g,
    '\n',
  );

  function numericColumns(model: string, type: 'Int' | 'Float'): string[] {
    const body = new RegExp(String.raw`model ${model} \{([\s\S]*?)\n\}`).exec(schema)?.[1];
    expect(body, `model ${model} not found`).toBeDefined();
    return [...body!.matchAll(new RegExp(String.raw`^\s+(\w+)\s+${type}\??(?:\s|$)`, 'gm'))]
      .map((m) => m[1]!)
      .sort();
  }

  it.each(Object.entries(MODEL_BY_KIND))('covers every numeric column of %s', (kind, model) => {
    const ints = (INT_FIELDS as Partial<Record<string, readonly string[]>>)[kind] ?? [];
    const floats = (FLOAT_FIELDS as Partial<Record<string, readonly string[]>>)[kind] ?? [];
    expect([...ints].sort()).toEqual(numericColumns(model, 'Int'));
    expect([...floats].sort()).toEqual(numericColumns(model, 'Float'));
  });
});
