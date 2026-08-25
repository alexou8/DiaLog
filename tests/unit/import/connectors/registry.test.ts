import { readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { parseCsv, parseXmlText } from '@/lib/import/parse';
import { CONNECTORS, detectConnector, getConnector } from '@/lib/import/connectors/registry';
import type { ParsedFile } from '@/lib/import/types';

function fixture(relPath: string, mimeType: string): ParsedFile {
  const text = readFileSync(path.join(process.cwd(), relPath), 'utf8');
  const filename = path.basename(relPath);
  if (mimeType === 'application/json') return { filename, mimeType, text, json: JSON.parse(text) };
  if (mimeType === 'application/xml') return { filename, mimeType, text, xml: parseXmlText(text) };
  return { filename, mimeType, text, rows: parseCsv(text) };
}

describe('CONNECTORS registry', () => {
  it('has unique connector ids', () => {
    const ids = CONNECTORS.map((c) => c.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it('includes generic-csv as the fallback', () => {
    expect(getConnector('generic-csv')).toBeDefined();
  });

  it('getConnector returns undefined for an unknown id', () => {
    expect(getConnector('nope')).toBeUndefined();
  });
});

describe('detectConnector', () => {
  it('picks dialog-legacy for the long-format sample log', () => {
    const file = fixture('ml/data/sample_logs.csv', 'text/csv');
    const result = detectConnector(file);
    expect(result?.connector.id).toBe('dialog-legacy');
  });

  it('picks dialog-legacy for the wide-format sample log', () => {
    const file = fixture('ml/data/sample_glucose_data.csv', 'text/csv');
    const result = detectConnector(file);
    expect(result?.connector.id).toBe('dialog-legacy');
  });

  it('picks abbott-libreview for a LibreView export', () => {
    const file = fixture('tests/fixtures/import/libreview.csv', 'text/csv');
    const result = detectConnector(file);
    expect(result?.connector.id).toBe('abbott-libreview');
  });

  it('picks nightscout for an entries.json export', () => {
    const file = fixture('tests/fixtures/import/nightscout-entries.json', 'application/json');
    const result = detectConnector(file);
    expect(result?.connector.id).toBe('nightscout');
  });

  it('picks apple-health for an export.xml', () => {
    const file = fixture('tests/fixtures/import/apple-health-export.xml', 'application/xml');
    const result = detectConnector(file);
    expect(result?.connector.id).toBe('apple-health');
  });

  it('picks omron for an Omron blood-pressure export', () => {
    const file = fixture('tests/fixtures/import/omron.csv', 'text/csv');
    const result = detectConnector(file);
    expect(result?.connector.id).toBe('omron');
  });

  it('falls back to generic-csv for an unrecognised but glucose-shaped CSV', () => {
    const file = fixture('tests/fixtures/import/rfc4180.csv', 'text/csv');
    const result = detectConnector(file);
    expect(result?.connector.id).toBe('generic-csv');
  });

  it('returns null when nothing can parse the file', () => {
    const file: ParsedFile = { filename: 'empty.csv', mimeType: 'text/csv', rows: [] };
    expect(detectConnector(file)).toBeNull();
  });
});
