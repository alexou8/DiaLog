import { readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { parseCsv } from '@/lib/import/parse';
import { abbottLibreViewConnector } from '@/lib/import/connectors/abbott-libreview';
import type { GlucoseRecord, MealRecord, ParsedFile } from '@/lib/import/types';

const NOW = new Date('2026-08-24T12:00:00Z');

function loadFixture(): ParsedFile {
  const text = readFileSync(
    path.join(process.cwd(), 'tests/fixtures/import/libreview.csv'),
    'utf8',
  );
  return { filename: 'libreview.csv', mimeType: 'text/csv', text, rows: parseCsv(text) };
}

describe('abbottLibreViewConnector', () => {
  it('detects the file past its preamble rows', async () => {
    expect(abbottLibreViewConnector.detect(loadFixture())).toBeGreaterThan(0.5);
  });

  it('does not detect an unrelated CSV', async () => {
    const file: ParsedFile = {
      filename: 'x.csv',
      mimeType: 'text/csv',
      rows: [
        ['a', 'b'],
        ['1', '2'],
      ],
    };
    expect(abbottLibreViewConnector.detect(file)).toBe(0);
  });

  it('parses historic and scan glucose rows in mg/dL', async () => {
    const result = await abbottLibreViewConnector.parse(loadFixture(), { now: NOW });
    expect(result.detectedUnit).toBe('MGDL');
    const glucoseRecords = result.records.filter((r): r is GlucoseRecord => r.kind === 'glucose');
    expect(glucoseRecords).toHaveLength(2);
    expect(glucoseRecords[0]?.valueMgdl).toBe(110);
    expect(glucoseRecords[1]?.valueMgdl).toBe(115);
  });

  it('produces a meal record with carbs for a Record Type 3 row', async () => {
    const result = await abbottLibreViewConnector.parse(loadFixture(), { now: NOW });
    const meal = result.records.find((r): r is MealRecord => r.kind === 'meal');
    expect(meal?.carbsG).toBe(45);
  });

  it('produces a medication record for a rapid-acting insulin row', async () => {
    const result = await abbottLibreViewConnector.parse(loadFixture(), { now: NOW });
    const med = result.records.find((r) => r.kind === 'medication');
    expect(med).toBeDefined();
  });

  it('reports no unsupported rows for a well-formed fixture', async () => {
    const result = await abbottLibreViewConnector.parse(loadFixture(), { now: NOW });
    expect(result.issues.filter((i) => i.code === 'UNSUPPORTED_ROW')).toHaveLength(0);
  });
});
