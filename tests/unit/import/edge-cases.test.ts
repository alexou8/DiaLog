import { describe, expect, it } from 'vitest';
import { parseCsv, parseFile, parseJsonText, parseXmlText } from '@/lib/import/parse';
import { genericCsvConnector } from '@/lib/import/connectors/generic-csv';
import { genericJsonConnector } from '@/lib/import/connectors/generic-json';
import { dedupeRecords } from '@/lib/import/dedupe';
import { buildImportSummary } from '@/lib/import/summary';

const NOW = new Date('2026-08-24T12:00:00Z');

describe('empty files', () => {
  it('an empty CSV file produces no records, no issues, and a warning', async () => {
    const result = await genericCsvConnector.parse(
      { filename: 'x.csv', mimeType: 'text/csv', rows: [] },
      { now: NOW },
    );
    expect(result.records).toHaveLength(0);
    expect(result.issues).toHaveLength(0);
    expect(result.warnings.length).toBeGreaterThan(0);
  });

  it('parseCsv on an empty string returns no rows', () => {
    expect(parseCsv('')).toEqual([]);
  });

  it('an empty JSON array produces no records and no issues', async () => {
    const result = await genericJsonConnector.parse(
      { filename: 'x.json', mimeType: 'application/json', json: [] },
      { now: NOW },
    );
    expect(result.records).toHaveLength(0);
    expect(result.issues).toHaveLength(0);
  });
});

describe('malformed / corrupt files', () => {
  it('throws a clear error for invalid JSON text rather than crashing silently', () => {
    expect(() => parseJsonText('{not valid json')).toThrow();
  });

  it('parses malformed XML leniently (fast-xml-parser is forgiving) without throwing', () => {
    expect(() => parseXmlText('<records><record type="glucose"</records>')).not.toThrow();

    // "Billion laughs": a few hundred bytes of nested DOCTYPE entities that
    // expand to gigabytes during parsing. The MAX_FILE_BYTES ceiling cannot
    // catch this because the blow-up happens after the size check passes, so
    // parseXmlText rejects any DOCTYPE internal subset outright.
    const bomb = [
      '<?xml version="1.0"?>',
      '<!DOCTYPE lolz [',
      '  <!ENTITY lol "lol">',
      '  <!ENTITY lol1 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">',
      '  <!ENTITY lol2 "&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;">',
      ']>',
      '<lolz>&lol2;</lolz>',
    ].join('\n');
    expect(() => parseXmlText(bomb)).toThrow(/DOCTYPE internal subset/);

    // A plain DOCTYPE with no internal subset defines no entities, so it is
    // harmless and must still parse.
    expect(() =>
      parseXmlText('<!DOCTYPE HealthData><HealthData><Record value="120"/></HealthData>'),
    ).not.toThrow();

    // Predefined entities must keep working -- Apple Health sourceName
    // attributes really do contain "&amp;".
    const parsed = parseXmlText(
      '<HealthData><Record sourceName="Dexcom &amp; Friends" value="120"/></HealthData>',
    ) as { HealthData: { Record: Record<string, string> } };
    expect(parsed.HealthData.Record['@_sourceName']).toBe('Dexcom & Friends');
  });

  it('a CSV with only a header row (no data rows) yields zero total rows', async () => {
    const result = await genericCsvConnector.parse(
      { filename: 'x.csv', mimeType: 'text/csv', rows: parseCsv('timestamp,glucose\n') },
      { now: NOW },
    );
    expect(result.rowsTotal).toBe(0);
    expect(result.records).toHaveLength(0);
  });

  it('rejects a JSON payload larger than the size guard', () => {
    const huge = JSON.stringify({ data: 'x'.repeat(60 * 1024 * 1024) });
    expect(() => parseJsonText(huge)).toThrow(/too large/);
  });
});

describe('a file where every row is rejected', () => {
  it('reports rowsTotal > 0 but zero imported records, all as issues', async () => {
    const csv = 'timestamp,glucose\nnot-a-date,120\n2026-01-10 08:00,not-a-number\n';
    const result = await genericCsvConnector.parse(
      { filename: 'x.csv', mimeType: 'text/csv', rows: parseCsv(csv) },
      { now: NOW, unit: 'MGDL' },
    );
    expect(result.rowsTotal).toBe(2);
    expect(result.records).toHaveLength(0);
    expect(result.issues).toHaveLength(2);

    const dedupe = dedupeRecords(result.records);
    const summary = buildImportSummary(result, dedupe);
    expect(summary.rowsImported).toBe(0);
    expect(summary.rowsRejected).toBe(2);
  });
});

describe('duplicate detection integration', () => {
  it('a file re-imported twice produces zero fresh records the second time', async () => {
    const csv = 'timestamp,glucose\n2026-01-10 08:00,120\n2026-01-10 09:00,130\n';
    const file = { filename: 'x.csv', mimeType: 'text/csv', rows: parseCsv(csv) };
    const first = await genericCsvConnector.parse(file, { now: NOW });
    const firstDedupe = dedupeRecords(first.records);
    expect(firstDedupe.fresh).toHaveLength(2);

    const existingKeys = new Set(firstDedupe.fresh.map((k) => k.dedupeKey));
    const second = await genericCsvConnector.parse(file, { now: NOW });
    const secondDedupe = dedupeRecords(second.records, existingKeys);
    expect(secondDedupe.fresh).toHaveLength(0);
    expect(secondDedupe.duplicates).toHaveLength(2);
  });
});

describe('end-to-end parseFile', () => {
  it('routes a .csv file through the CSV parser', async () => {
    const parsed = await parseFile('x.csv', 'text/csv', 'a,b\n1,2\n');
    expect(parsed.rows).toEqual([
      ['a', 'b'],
      ['1', '2'],
    ]);
  });

  it('routes a .json file through the JSON parser', async () => {
    const parsed = await parseFile('x.json', 'application/json', '[{"a":1}]');
    expect(parsed.json).toEqual([{ a: 1 }]);
  });

  it('routes a .xml file through the XML parser', async () => {
    const parsed = await parseFile('x.xml', 'application/xml', '<a><b>1</b></a>');
    expect(parsed.xml).toBeDefined();
  });

  it('rejects a file larger than the max size guard', async () => {
    const big = Buffer.alloc(101 * 1024 * 1024, 'a');
    await expect(parseFile('big.csv', 'text/csv', big)).rejects.toThrow(/too large/);
  });
});
