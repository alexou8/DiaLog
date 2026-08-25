import { readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { parseCsv } from '@/lib/import/parse';
import { omronConnector } from '@/lib/import/connectors/omron';
import type { NoteRecord, ParsedFile } from '@/lib/import/types';

const NOW = new Date('2026-08-24T12:00:00Z');

function loadFixture(): ParsedFile {
  const text = readFileSync(path.join(process.cwd(), 'tests/fixtures/import/omron.csv'), 'utf8');
  return { filename: 'omron.csv', mimeType: 'text/csv', text, rows: parseCsv(text) };
}

describe('omronConnector', () => {
  it('detects a blood-pressure export', () => {
    expect(omronConnector.detect(loadFixture())).toBeGreaterThan(0);
  });

  it('does not detect an unrelated CSV', () => {
    const file: ParsedFile = {
      filename: 'x.csv',
      mimeType: 'text/csv',
      rows: [
        ['a', 'b'],
        ['1', '2'],
      ],
    };
    expect(omronConnector.detect(file)).toBe(0);
  });

  it('parses systolic/diastolic/pulse into a note record', async () => {
    const result = await omronConnector.parse(loadFixture(), { now: NOW, dateOrder: 'DMY' });
    expect(result.issues).toHaveLength(0);
    expect(result.records).toHaveLength(2);
    const note = result.records[0] as NoteRecord;
    expect(note.kind).toBe('note');
    expect(note.text).toContain('120/80');
  });

  it('parses a weight column into a weight record', async () => {
    const file: ParsedFile = {
      filename: 'weight.csv',
      mimeType: 'text/csv',
      rows: parseCsv('Date,Time,Weight (kg)\n20/01/2026,08:00,72.4\n'),
    };
    const result = await omronConnector.parse(file, { now: NOW, dateOrder: 'DMY' });
    expect(result.records).toHaveLength(1);
    expect(result.records[0]?.kind).toBe('weight');
  });

  it('flags an unparseable timestamp', async () => {
    const file: ParsedFile = {
      filename: 'x.csv',
      mimeType: 'text/csv',
      rows: parseCsv('Date,Time,SYS(mmHg),DIA(mmHg)\nnot-a-date,08:00,120,80\n'),
    };
    const result = await omronConnector.parse(file, { now: NOW });
    expect(result.issues[0]?.code).toBe('INVALID_TIMESTAMP');
  });
});
