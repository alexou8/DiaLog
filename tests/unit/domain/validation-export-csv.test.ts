/**
 * Output validation for the CSV export: formula injection and RFC4180 quoting.
 *
 * A note or meal description — often carried in from a third-party import —
 * that starts with `=`, `+`, `-`, `@`, tab or CR runs as a formula when the
 * export is opened in a spreadsheet. The database-backed behaviour (which
 * columns are neutralised, that numbers are not) is covered in
 * tests/integration/export-complete.test.ts; these pin the pure helpers.
 */
import { describe, expect, it, vi } from 'vitest';

// The helpers are pure; the module just happens to import the Prisma client.
vi.mock('@/lib/db/prisma', () => ({ prisma: {} }));

import { csvField, csvRow, neutraliseFormula } from '@/lib/services/export-service';

describe('neutraliseFormula', () => {
  it.each([
    ['=HYPERLINK("https://evil.example","Open")', `'=HYPERLINK("https://evil.example","Open")`],
    ['+1+1', "'+1+1"],
    ['-2+3', "'-2+3"],
    ['@SUM(A1:A2)', "'@SUM(A1:A2)"],
    ['\t=1+1', "'\t=1+1"],
    ['\r=1+1', "'\r=1+1"],
  ])('prefixes %j', (input, expected) => {
    expect(neutraliseFormula(input)).toBe(expected);
  });

  it.each(['pizza, 2 slices', 'felt fine', '', ' =not at the start', '5 units'])(
    'leaves ordinary text %j alone',
    (input) => {
      expect(neutraliseFormula(input)).toBe(input);
    },
  );
});

describe('csvField / csvRow', () => {
  it('quotes commas, quotes and line breaks, doubling embedded quotes', () => {
    expect(csvField('a,b')).toBe('"a,b"');
    expect(csvField('say "hi"')).toBe('"say ""hi"""');
    expect(csvField('line\nbreak')).toBe('"line\nbreak"');
  });

  it('writes numbers as numbers, so a negative value is not turned into text', () => {
    expect(csvRow([-5, 0, 7.5, null, undefined])).toBe('-5,0,7.5,,');
  });

  it('still quotes a neutralised cell that needs quoting', () => {
    expect(csvField(neutraliseFormula('=A1,B1'))).toBe(`"'=A1,B1"`);
  });
});
