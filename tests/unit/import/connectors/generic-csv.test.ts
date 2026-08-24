import { describe, expect, it } from 'vitest';
import { parseCsv } from '@/lib/import/parse';
import { genericCsvConnector } from '@/lib/import/connectors/generic-csv';
import type { GlucoseRecord, ParsedFile } from '@/lib/import/types';

function fileFromCsv(text: string, filename = 'export.csv'): ParsedFile {
  return { filename, mimeType: 'text/csv', text, rows: parseCsv(text) };
}

const NOW = new Date('2026-08-24T12:00:00Z');

describe('genericCsvConnector.detect', () => {
  it('detects a file with a glucose column and a timestamp column', async () => {
    const file = fileFromCsv('timestamp,glucose\n2026-01-10 08:00,120\n');
    expect(genericCsvConnector.detect(file)).toBeGreaterThan(0);
  });

  it('does not detect a file with no glucose-like column', async () => {
    const file = fileFromCsv('date,weight\n2026-01-10,70\n');
    expect(genericCsvConnector.detect(file)).toBe(0);
  });

  it('does not detect an empty file', async () => {
    expect(genericCsvConnector.detect({ filename: 'x.csv', mimeType: 'text/csv', rows: [] })).toBe(0);
  });
});

describe('genericCsvConnector.parse', () => {
  it('parses a combined timestamp column with mg/dL values', async () => {
    const file = fileFromCsv('timestamp,glucose (mg/dL),notes\n2026-01-10 08:00,120,fasting\n');
    const result = await genericCsvConnector.parse(file, { now: NOW });
    expect(result.issues).toHaveLength(0);
    expect(result.records).toHaveLength(1);
    const rec = result.records[0] as GlucoseRecord;
    expect(rec.kind).toBe('glucose');
    expect(rec.valueMgdl).toBe(120);
    expect(rec.context).toBe('FASTING');
  });

  it('parses separate date + time columns', async () => {
    const file = fileFromCsv('date,time,bg\n2026-01-10,08:00,120\n');
    const result = await genericCsvConnector.parse(file, { now: NOW });
    expect(result.records).toHaveLength(1);
    expect((result.records[0] as GlucoseRecord).takenAt.toISOString()).toBe('2026-01-10T08:00:00.000Z');
  });

  it('infers mmol/L from value magnitude when no unit column is present', async () => {
    const file = fileFromCsv('timestamp,glucose\n2026-01-10 08:00,7.2\n2026-01-10 09:00,6.5\n');
    const result = await genericCsvConnector.parse(file, { now: NOW });
    expect(result.detectedUnit).toBe('MMOLL');
    expect(result.warnings.some((w) => w.includes('inferred'))).toBe(true);
    const rec = result.records[0] as GlucoseRecord;
    expect(rec.valueMgdl).toBeCloseTo(129.7, 0);
  });

  it('respects an explicit unit column value per row', async () => {
    const file = fileFromCsv('timestamp,glucose,unit\n2026-01-10 08:00,120,mg/dL\n2026-01-10 09:00,7.2,mmol/L\n');
    const result = await genericCsvConnector.parse(file, { now: NOW });
    expect(result.issues).toHaveLength(0);
    expect((result.records[0] as GlucoseRecord).valueMgdl).toBe(120);
    expect((result.records[1] as GlucoseRecord).valueMgdl).toBeCloseTo(129.7, 0);
  });

  it('rejects out-of-range values as OUT_OF_RANGE issues', async () => {
    const file = fileFromCsv('timestamp,glucose\n2026-01-10 08:00,5000\n');
    const result = await genericCsvConnector.parse(file, { now: NOW, unit: 'MGDL' });
    expect(result.records).toHaveLength(0);
    expect(result.issues[0]?.code).toBe('OUT_OF_RANGE');
  });

  it('flags a missing glucose value as MISSING_VALUE', async () => {
    const file = fileFromCsv('timestamp,glucose\n2026-01-10 08:00,\n');
    const result = await genericCsvConnector.parse(file, { now: NOW });
    expect(result.issues[0]?.code).toBe('MISSING_VALUE');
  });

  it('flags an invalid timestamp', async () => {
    const file = fileFromCsv('timestamp,glucose\nnot-a-date,120\n');
    const result = await genericCsvConnector.parse(file, { now: NOW, unit: 'MGDL' });
    expect(result.issues[0]?.code).toBe('INVALID_TIMESTAMP');
  });

  it('flags a future timestamp', async () => {
    const file = fileFromCsv('timestamp,glucose\n2026-08-30 08:00,120\n');
    const result = await genericCsvConnector.parse(file, { now: NOW, unit: 'MGDL' });
    expect(result.issues[0]?.code).toBe('FUTURE_TIMESTAMP');
  });

  it('also emits a companion meal record when a carbs column has a value', async () => {
    const file = fileFromCsv('timestamp,glucose,carbs (g)\n2026-01-10 08:00,120,45\n');
    const result = await genericCsvConnector.parse(file, { now: NOW });
    expect(result.records).toHaveLength(2);
    expect(result.records.some((r) => r.kind === 'meal')).toBe(true);
  });

  it('reports an empty file as a warning with no records or issues', async () => {
    const result = await genericCsvConnector.parse({ filename: 'x.csv', mimeType: 'text/csv', rows: [] }, { now: NOW });
    expect(result.records).toHaveLength(0);
    expect(result.warnings.length).toBeGreaterThan(0);
  });

  it('rejects every row as UNSUPPORTED_ROW when no glucose column exists', async () => {
    const file = fileFromCsv('date,weight\n2026-01-10,70\n2026-01-11,71\n');
    const result = await genericCsvConnector.parse(file, { now: NOW });
    expect(result.records).toHaveLength(0);
    expect(result.issues).toHaveLength(2);
    expect(result.issues.every((i) => i.code === 'UNSUPPORTED_ROW')).toBe(true);
  });
});
