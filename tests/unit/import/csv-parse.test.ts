import { describe, expect, it } from 'vitest';
import { detectDelimiter, parseCsv, stripBom } from '@/lib/import/parse';

describe('stripBom', () => {
  it('removes a leading UTF-8 BOM', () => {
    expect(stripBom('﻿hello')).toBe('hello');
  });
  it('leaves text without a BOM untouched', () => {
    expect(stripBom('hello')).toBe('hello');
  });
});

describe('detectDelimiter', () => {
  it('detects comma', () => {
    expect(detectDelimiter('a,b,c\n1,2,3\n')).toBe(',');
  });
  it('detects semicolon', () => {
    expect(detectDelimiter('a;b;c\n1;2;3\n')).toBe(';');
  });
  it('detects tab', () => {
    expect(detectDelimiter('a\tb\tc\n1\t2\t3\n')).toBe('\t');
  });
  it('detects pipe', () => {
    expect(detectDelimiter('a|b|c\n1|2|3\n')).toBe('|');
  });
});

describe('parseCsv (RFC4180)', () => {
  it('parses plain fields', () => {
    const rows = parseCsv('a,b,c\n1,2,3\n');
    expect(rows).toEqual([
      ['a', 'b', 'c'],
      ['1', '2', '3'],
    ]);
  });

  it('handles quoted fields with embedded commas', () => {
    const rows = parseCsv('name,note\nA,"hello, world"\n');
    expect(rows).toEqual([
      ['name', 'note'],
      ['A', 'hello, world'],
    ]);
  });

  it('handles escaped double quotes inside quoted fields', () => {
    const rows = parseCsv('name,note\nA,"she said ""hi"""\n');
    expect(rows[1]).toEqual(['A', 'she said "hi"']);
  });

  it('handles embedded newlines inside quoted fields', () => {
    const rows = parseCsv('name,note\nA,"line one\nline two"\n');
    expect(rows).toEqual([
      ['name', 'note'],
      ['A', 'line one\nline two'],
    ]);
  });

  it('strips a BOM before parsing', () => {
    const rows = parseCsv('﻿a,b\n1,2\n');
    expect(rows[0]).toEqual(['a', 'b']);
  });

  it('auto-detects semicolon delimiter', () => {
    const rows = parseCsv('a;b\n1;2\n');
    expect(rows).toEqual([
      ['a', 'b'],
      ['1', '2'],
    ]);
  });

  it('handles a file with no trailing newline', () => {
    const rows = parseCsv('a,b\n1,2');
    expect(rows).toEqual([
      ['a', 'b'],
      ['1', '2'],
    ]);
  });

  it('handles CRLF line endings', () => {
    const rows = parseCsv('a,b\r\n1,2\r\n');
    expect(rows).toEqual([
      ['a', 'b'],
      ['1', '2'],
    ]);
  });

  it('returns an empty array for an empty string', () => {
    expect(parseCsv('')).toEqual([]);
  });

  it('preserves ragged (short) rows rather than throwing', () => {
    const rows = parseCsv('a,b,c\n1,2\n');
    expect(rows[1]).toEqual(['1', '2']);
  });
});
