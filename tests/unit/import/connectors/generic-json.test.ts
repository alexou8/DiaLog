import { readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { genericJsonConnector } from '@/lib/import/connectors/generic-json';
import type { GlucoseRecord, MealRecord, MedicationRecord, ParsedFile } from '@/lib/import/types';

const NOW = new Date('2026-08-24T12:00:00Z');

function loadFixture(): ParsedFile {
  const text = readFileSync(path.join(process.cwd(), 'tests/fixtures/import/generic.json'), 'utf8');
  return { filename: 'generic.json', mimeType: 'application/json', text, json: JSON.parse(text) };
}

describe('genericJsonConnector', () => {
  it('detects an array of typed records', () => {
    expect(genericJsonConnector.detect(loadFixture())).toBeGreaterThan(0);
  });

  it('does not detect an array of untyped objects', () => {
    const file: ParsedFile = { filename: 'x.json', mimeType: 'application/json', json: [{ foo: 'bar' }] };
    expect(genericJsonConnector.detect(file)).toBe(0);
  });

  it('parses glucose, meal, and medication records', async () => {
    const result = await genericJsonConnector.parse(loadFixture(), { now: NOW });
    expect(result.issues).toHaveLength(0);
    expect(result.records).toHaveLength(3);
    const glucose = result.records.find((r): r is GlucoseRecord => r.kind === 'glucose');
    expect(glucose?.valueMgdl).toBe(120);
    expect(glucose?.context).toBe('FASTING');
    const meal = result.records.find((r): r is MealRecord => r.kind === 'meal');
    expect(meal?.carbsG).toBe(45);
    const med = result.records.find((r): r is MedicationRecord => r.kind === 'medication');
    expect(med?.dose).toBe('500mg');
  });

  it('flags an object with an unrecognised type as UNSUPPORTED_ROW', async () => {
    const file: ParsedFile = {
      filename: 'x.json',
      mimeType: 'application/json',
      json: [{ type: 'bloodpressure', takenAt: '2026-01-10T08:00:00Z' }],
    };
    const result = await genericJsonConnector.parse(file, { now: NOW });
    expect(result.issues[0]?.code).toBe('UNSUPPORTED_ROW');
  });

  it('flags a missing timestamp', async () => {
    const file: ParsedFile = { filename: 'x.json', mimeType: 'application/json', json: [{ type: 'glucose', value: 100 }] };
    const result = await genericJsonConnector.parse(file, { now: NOW });
    expect(result.issues[0]?.code).toBe('MISSING_TIMESTAMP');
  });

  it('supports the {records:[...]} wrapper shape', async () => {
    const file: ParsedFile = {
      filename: 'x.json',
      mimeType: 'application/json',
      json: { records: [{ type: 'weight', takenAt: '2026-01-10T08:00:00Z', weightKg: 70 }] },
    };
    const result = await genericJsonConnector.parse(file, { now: NOW });
    expect(result.records).toHaveLength(1);
    expect(result.records[0]?.kind).toBe('weight');
  });

  it('reports a non-array JSON payload as a warning with no records', async () => {
    const file: ParsedFile = { filename: 'x.json', mimeType: 'application/json', json: { foo: 'bar' } };
    const result = await genericJsonConnector.parse(file, { now: NOW });
    expect(result.records).toHaveLength(0);
    expect(result.warnings.length).toBeGreaterThan(0);
  });
});
