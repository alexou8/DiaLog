import { readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { parseXmlText } from '@/lib/import/parse';
import { genericXmlConnector } from '@/lib/import/connectors/generic-xml';
import type { GlucoseRecord, MealRecord, ParsedFile } from '@/lib/import/types';

const NOW = new Date('2026-08-24T12:00:00Z');

function loadFixture(): ParsedFile {
  const text = readFileSync(path.join(process.cwd(), 'tests/fixtures/import/generic.xml'), 'utf8');
  return { filename: 'generic.xml', mimeType: 'application/xml', text, xml: parseXmlText(text) };
}

describe('genericXmlConnector', () => {
  it('detects a <records><record type=.../></records> document', () => {
    expect(genericXmlConnector.detect(loadFixture())).toBeGreaterThan(0);
  });

  it('does not detect an unrelated XML document', () => {
    const file: ParsedFile = {
      filename: 'x.xml',
      mimeType: 'application/xml',
      xml: { foo: { bar: 'baz' } },
    };
    expect(genericXmlConnector.detect(file)).toBe(0);
  });

  it('parses glucose and meal records with attributes', async () => {
    const result = await genericXmlConnector.parse(loadFixture(), { now: NOW });
    expect(result.issues).toHaveLength(0);
    expect(result.records).toHaveLength(2);
    const glucose = result.records.find((r): r is GlucoseRecord => r.kind === 'glucose');
    expect(glucose?.valueMgdl).toBe(120);
    expect(glucose?.context).toBe('FASTING');
    const meal = result.records.find((r): r is MealRecord => r.kind === 'meal');
    expect(meal?.carbsG).toBe(45);
  });

  it('handles a single <record> element (not wrapped in an array)', async () => {
    const text =
      '<records><record type="weight" takenAt="2026-01-10T08:00:00Z" weightKg="70" /></records>';
    const file: ParsedFile = {
      filename: 'x.xml',
      mimeType: 'application/xml',
      text,
      xml: parseXmlText(text),
    };
    const result = await genericXmlConnector.parse(file, { now: NOW });
    expect(result.records).toHaveLength(1);
    expect(result.records[0]?.kind).toBe('weight');
  });

  it('flags an unrecognised record type', async () => {
    const text = '<records><record type="unknown" takenAt="2026-01-10T08:00:00Z" /></records>';
    const file: ParsedFile = {
      filename: 'x.xml',
      mimeType: 'application/xml',
      text,
      xml: parseXmlText(text),
    };
    const result = await genericXmlConnector.parse(file, { now: NOW });
    expect(result.issues[0]?.code).toBe('UNSUPPORTED_ROW');
  });

  it('reports a document with no <record> elements as a warning', async () => {
    const text = '<foo><bar/></foo>';
    const file: ParsedFile = {
      filename: 'x.xml',
      mimeType: 'application/xml',
      text,
      xml: parseXmlText(text),
    };
    const result = await genericXmlConnector.parse(file, { now: NOW });
    expect(result.records).toHaveLength(0);
    expect(result.warnings.length).toBeGreaterThan(0);
  });
});
