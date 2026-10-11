import { readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { parseCsv } from '@/lib/import/parse';
import { abbottOptiumNeoConnector } from '@/lib/import/connectors/abbott-optium-neo';
import { genericCsvConnector } from '@/lib/import/connectors/generic-csv';
import { detectConnector } from '@/lib/import/connectors/registry';
import type { GlucoseRecord, ParsedFile } from '@/lib/import/types';

const NOW = new Date('2026-08-24T12:00:00Z');

function loadFixture(): ParsedFile {
  const text = readFileSync(
    path.join(process.cwd(), 'tests/fixtures/import/abbott-optium-neo.csv'),
    'utf8',
  );
  return { filename: 'optium.csv', mimeType: 'text/csv', rows: parseCsv(text) };
}

describe('abbottOptiumNeoConnector', () => {
  it('detects the Abbott report preamble', () => {
    expect(abbottOptiumNeoConnector.detect(loadFixture())).toBeGreaterThan(0);
  });

  it('wins detection over generic-csv for an Abbott export', () => {
    expect(detectConnector(loadFixture())?.connector.id).toBe('abbott-optium-neo');
    // generic-csv alone sees the preamble line as the header and gives up.
    expect(genericCsvConnector.detect(loadFixture())).toBe(0);
  });

  it('does not detect a CSV without the Abbott preamble', () => {
    const file: ParsedFile = {
      filename: 'x.csv',
      mimeType: 'text/csv',
      rows: parseCsv('timestamp,glucose\n2026-01-10 08:00,110\n'),
    };
    expect(abbottOptiumNeoConnector.detect(file)).toBe(0);
  });

  it('skips the preamble, reads the "Result" column, and converts mmol/L', async () => {
    const result = await abbottOptiumNeoConnector.parse(loadFixture(), {
      now: NOW,
      timezone: 'UTC',
    });
    const glucose = result.records.filter((r): r is GlucoseRecord => r.kind === 'glucose');
    expect(glucose).toHaveLength(5);
    // 13/01/2026 proves day-first; 5.8 mmol/L is ~104.5 mg/dL canonical.
    expect(glucose[0]?.takenAt.toISOString()).toBe('2026-01-13T07:42:00.000Z');
    expect(glucose[0]?.valueMgdl).toBeCloseTo(5.8 * 18.0182, 2);
    expect(glucose[0]?.context).toBe('BEFORE_MEAL');
    expect(result.warnings.some((w) => /Skipped 4 lines/.test(w))).toBe(true);
  });

  it('never silently drops a row: every data row is a record or an issue', async () => {
    const result = await abbottOptiumNeoConnector.parse(loadFixture(), { now: NOW });
    expect(result.rowsTotal).toBe(9);
    expect(result.records.length + result.issues.length).toBe(result.rowsTotal);
    expect(result.issues.map((i) => i.code).sort()).toEqual([
      'INVALID_TIMESTAMP', // 25:00
      'INVALID_VALUE', // "HI"
      'MISSING_VALUE', // empty result
      'UNSUPPORTED_ROW', // blood ketone
    ]);
  });

  it('refuses a blood ketone result rather than reading it as glucose', async () => {
    // 1.5 mmol/L ketone would otherwise be a plausible-looking 1.5 mmol/L glucose.
    const result = await abbottOptiumNeoConnector.parse(loadFixture(), { now: NOW });
    const ketone = result.issues.find((i) => i.code === 'UNSUPPORTED_ROW');
    expect(ketone?.rawRow).toMatch(/Ketone/);
    // Row numbers count from the top of the parsed file, preamble included.
    expect(ketone?.rowNumber).toBe(11);
  });
});
