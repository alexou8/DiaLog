/**
 * Real Apple Health export.xml files carry an inline DTD of ELEMENT/ATTLIST
 * declarations, so "reject any internal subset" rejected every real Apple
 * Health import. The rule is now: accept an internal subset only when it
 * declares no entities and names no external DTD, then strip it before the
 * parser runs. These tests pin both halves.
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { ImportLimitError } from '@/lib/import/limits';
import { parseFile, parseXmlText, stripSafeDoctype } from '@/lib/import/parse';

const BODY = '<HealthData locale="en_US"><Record type="t" value="120"/></HealthData>';

function rejectsAsUnsafe(xml: string): void {
  let error: unknown;
  try {
    parseXmlText(xml);
  } catch (e) {
    error = e;
  }
  expect(error).toBeInstanceOf(ImportLimitError);
  expect((error as ImportLimitError).reason).toBe('UNSAFE_XML');
}

describe('accepted DOCTYPEs', () => {
  it('parses the Apple Health fixture, whose DOCTYPE has ELEMENT/ATTLIST declarations', () => {
    const text = readFileSync(
      path.join(process.cwd(), 'tests/fixtures/import/apple-health-export.xml'),
      'utf8',
    );
    expect(text).toMatch(/<!DOCTYPE HealthData \[/);
    expect(text).toMatch(/<!ATTLIST Record/);
    const parsed = parseXmlText(text) as { HealthData: { Record: unknown[] } };
    expect(parsed.HealthData.Record).toHaveLength(5);
  });

  it('accepts a DTD whose comments or quoted defaults mention > and ]', () => {
    const xml = [
      '<?xml version="1.0" encoding="UTF-8"?>',
      '<!-- exported -->',
      '<!DOCTYPE HealthData [',
      '<!-- HealthKit Export Version: 13 > ] -->',
      '<!ELEMENT HealthData (Record)*>',
      '<!ATTLIST Record unit CDATA "a>b]">',
      ']>',
      BODY,
    ].join('\n');
    const parsed = parseXmlText(xml) as { HealthData: { Record: Record<string, string> } };
    expect(parsed.HealthData.Record['@_value']).toBe('120');
  });

  it('accepts a bare DOCTYPE with no subset', () => {
    expect(() => parseXmlText(`<!DOCTYPE HealthData>${BODY}`)).not.toThrow();
  });

  it('removes the DOCTYPE but keeps line numbers aligned', () => {
    const xml = `<!DOCTYPE r [\n<!ELEMENT r ANY>\n]>\n<r/>`;
    const stripped = stripSafeDoctype(xml);
    expect(stripped).not.toMatch(/DOCTYPE/);
    expect(stripped.split('\n')).toHaveLength(xml.split('\n').length);
  });

  it('keeps decoding the predefined entities (&amp; in Apple sourceName)', () => {
    const parsed = parseXmlText(
      '<!DOCTYPE HealthData [<!ELEMENT HealthData ANY>]><HealthData><Record sourceName="Dexcom &amp; Friends"/></HealthData>',
    ) as { HealthData: { Record: Record<string, string> } };
    expect(parsed.HealthData.Record['@_sourceName']).toBe('Dexcom & Friends');
  });
});

describe('rejected DOCTYPEs', () => {
  it('rejects an internal subset that declares an entity', () => {
    rejectsAsUnsafe(`<!DOCTYPE HealthData [<!ELEMENT HealthData ANY><!ENTITY x "y">]>${BODY}`);
    rejectsAsUnsafe(`<!DOCTYPE HealthData [<!entity x "y">]>${BODY}`); // case-insensitive
  });

  it('rejects the billion-laughs bomb', () => {
    const bomb = [
      '<?xml version="1.0"?>',
      '<!DOCTYPE lolz [',
      '  <!ENTITY lol "lol">',
      '  <!ENTITY lol1 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">',
      '  <!ENTITY lol2 "&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;">',
      ']>',
      '<lolz>&lol2;</lolz>',
    ].join('\n');
    rejectsAsUnsafe(bomb);
  });

  it('rejects the quoted-">" trick that the old regex missed', () => {
    // /<!DOCTYPE[^>[]*\[/ stopped at the ">" inside the quotes and never saw
    // the subset; the parser, which tracks quotes, did.
    rejectsAsUnsafe(`<!DOCTYPE r SYSTEM "a>b" [<!ENTITY e "boom">]><r>&e;</r>`);
    rejectsAsUnsafe(`<!DOCTYPE r [<!ATTLIST r a CDATA "x>y"><!ENTITY e "boom">]><r>&e;</r>`);
  });

  it('rejects an external DTD reference even without a subset', () => {
    rejectsAsUnsafe(`<!DOCTYPE r SYSTEM "http://example.invalid/x.dtd"><r/>`);
    rejectsAsUnsafe(`<!DOCTYPE r PUBLIC "-//X//DTD X//EN" "x.dtd"><r/>`);
  });

  it('rejects an entity hidden after a comment-like decoy', () => {
    rejectsAsUnsafe(`<!DOCTYPE r [<!-- <!ELEMENT r ANY> --><!ENTITY e "boom">]><r>&e;</r>`);
  });

  it('rejects a DOCTYPE or ENTITY that appears outside the prolog', () => {
    rejectsAsUnsafe(`<r/><!DOCTYPE r [<!ENTITY e "boom">]>`);
    rejectsAsUnsafe(`garbage <!DOCTYPE r [<!ENTITY e "boom">]><r>&e;</r>`);
  });

  it('rejects an unterminated DOCTYPE', () => {
    rejectsAsUnsafe(`<!DOCTYPE r [<!ELEMENT r ANY>`);
  });

  it('surfaces the typed error through parseFile', async () => {
    await expect(
      parseFile('export.xml', 'application/xml', `<!DOCTYPE r [<!ENTITY e "x">]><r/>`),
    ).rejects.toBeInstanceOf(ImportLimitError);
  });
});
