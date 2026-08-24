import { readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { nightscoutConnector } from '@/lib/import/connectors/nightscout';
import type { GlucoseRecord, ParsedFile } from '@/lib/import/types';

const NOW = new Date('2026-08-24T12:00:00Z');

function loadFixture(): ParsedFile {
  const text = readFileSync(path.join(process.cwd(), 'tests/fixtures/import/nightscout-entries.json'), 'utf8');
  return { filename: 'entries.json', mimeType: 'application/json', text, json: JSON.parse(text) };
}

describe('nightscoutConnector', () => {
  it('detects a Nightscout entries array', () => {
    expect(nightscoutConnector.detect(loadFixture())).toBeGreaterThan(0);
  });

  it('does not detect an unrelated JSON array', () => {
    const file: ParsedFile = { filename: 'x.json', mimeType: 'application/json', json: [{ foo: 'bar' }] };
    expect(nightscoutConnector.detect(file)).toBe(0);
  });

  it('parses sgv and mbg entries as mg/dL glucose readings', async () => {
    const result = await nightscoutConnector.parse(loadFixture(), { now: NOW });
    expect(result.issues).toHaveLength(0);
    expect(result.detectedUnit).toBe('MGDL');
    expect(result.records).toHaveLength(2);
    const sgv = result.records[0] as GlucoseRecord;
    expect(sgv.valueMgdl).toBe(145);
    expect(sgv.externalId).toBe('a1');
    expect(sgv.note).toContain('Flat');
    const mbg = result.records[1] as GlucoseRecord;
    expect(mbg.valueMgdl).toBe(150);
    expect(mbg.context).toBe('RANDOM');
  });

  it('flags an unrecognised entry type', async () => {
    const file: ParsedFile = {
      filename: 'x.json',
      mimeType: 'application/json',
      json: [{ type: 'cal', date: 1768032000000 }],
    };
    const result = await nightscoutConnector.parse(file, { now: NOW });
    expect(result.issues[0]?.code).toBe('UNSUPPORTED_ROW');
  });

  it('falls back to numeric `date` (epoch ms) when dateString is absent', async () => {
    const file: ParsedFile = {
      filename: 'x.json',
      mimeType: 'application/json',
      json: [{ type: 'sgv', sgv: 130, date: 1768032000000 }],
    };
    const result = await nightscoutConnector.parse(file, { now: NOW });
    expect(result.records).toHaveLength(1);
    expect((result.records[0] as GlucoseRecord).takenAt.toISOString()).toBe('2026-01-10T08:00:00.000Z');
  });
});
