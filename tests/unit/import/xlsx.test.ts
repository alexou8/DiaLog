/**
 * XLSX is a zip, so the upload ceiling only bounds compressed bytes. These
 * tests pin the two budgets that stop a small upload from inflating into
 * gigabytes: the bounded inflate of every zip entry, and the cell/string
 * budget while streaming the worksheet.
 */
import ExcelJS from 'exceljs';
import { describe, expect, it } from 'vitest';
import { ImportLimitError } from '@/lib/import/limits';
import { assertXlsxWithinBudget, parseFile, parseXlsx } from '@/lib/import/parse';

async function workbook(fill: (sheet: ExcelJS.Worksheet) => void): Promise<Buffer> {
  const wb = new ExcelJS.Workbook();
  fill(wb.addWorksheet('Readings'));
  return Buffer.from(await wb.xlsx.writeBuffer());
}

describe('parseXlsx', () => {
  it('reads the first worksheet into string rows, including dates', async () => {
    const buf = await workbook((sheet) => {
      sheet.addRow(['timestamp', 'glucose']);
      sheet.addRow([new Date('2026-01-10T08:00:00Z'), 110]);
      sheet.addRow([new Date('2026-01-10T09:00:00Z'), 'n/a']);
    });
    const rows = await parseXlsx(buf);
    expect(rows).toEqual([
      ['timestamp', 'glucose'],
      ['2026-01-10T08:00:00.000Z', '110'],
      ['2026-01-10T09:00:00.000Z', 'n/a'],
    ]);
  });

  it('is reached through parseFile for an .xlsx upload', async () => {
    const buf = await workbook((sheet) => {
      sheet.addRow(['timestamp', 'glucose']);
      sheet.addRow(['2026-01-10 08:00', 110]);
    });
    const parsed = await parseFile('readings.xlsx', '', buf);
    expect(parsed.rows?.[1]).toEqual(['2026-01-10 08:00', '110']);
  });

  it('refuses a workbook past the cell budget with a typed error', async () => {
    const buf = await workbook((sheet) => {
      for (let r = 0; r < 60; r++) sheet.addRow([`2026-01-10 08:${r}`, 100 + r, 'note']);
    });
    const error = await parseXlsx(buf, { maxCells: 100 }).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ImportLimitError);
    expect((error as ImportLimitError).reason).toBe('WORKBOOK_TOO_LARGE');
  });

  it('counts cells before ExcelJS builds its model', async () => {
    const buf = await workbook((sheet) => {
      for (let r = 0; r < 60; r++) sheet.addRow([r, r, r]);
    });
    expect(() =>
      assertXlsxWithinBudget(buf, { maxUncompressedBytes: 1e9, maxCells: 100, maxEntries: 100 }),
    ).toThrow(ImportLimitError);
    expect(() =>
      assertXlsxWithinBudget(buf, { maxUncompressedBytes: 1e9, maxCells: 180, maxEntries: 100 }),
    ).not.toThrow();
  });

  it('bounds text amplified through one shared string referenced by many cells', async () => {
    const shared = 'y'.repeat(50 * 1024);
    const buf = await workbook((sheet) => {
      for (let r = 0; r < 100; r++) sheet.addRow([shared]);
    });
    // The zip itself inflates to well under 1 MB; the text it stands for is ~5 MB.
    expect(() =>
      assertXlsxWithinBudget(buf, {
        maxUncompressedBytes: 1024 * 1024,
        maxCells: 1e6,
        maxEntries: 100,
      }),
    ).not.toThrow();
    await expect(parseXlsx(buf, { maxUncompressedBytes: 1024 * 1024 })).rejects.toBeInstanceOf(
      ImportLimitError,
    );
  });

  it('refuses a workbook whose zip inflates past the byte budget before ExcelJS reads it', async () => {
    // ~2 MB of one repeated character compresses to a few KB: a miniature
    // zip bomb. The budget is lowered so the test stays fast.
    const buf = await workbook((sheet) => {
      sheet.addRow(['x'.repeat(2 * 1024 * 1024)]);
    });
    expect(buf.length).toBeLessThan(100 * 1024);
    expect(() =>
      assertXlsxWithinBudget(buf, {
        maxUncompressedBytes: 1024 * 1024,
        maxCells: 1e6,
        maxEntries: 100,
      }),
    ).toThrow(ImportLimitError);
    await expect(parseXlsx(buf, { maxUncompressedBytes: 1024 * 1024 })).rejects.toThrow(
      /expands to more data/,
    );
  });

  it('refuses a workbook with more zip entries than any real spreadsheet has', async () => {
    const buf = await workbook((sheet) => sheet.addRow(['a']));
    expect(() =>
      assertXlsxWithinBudget(buf, { maxUncompressedBytes: 1e9, maxCells: 1e6, maxEntries: 2 }),
    ).toThrow(ImportLimitError);
  });

  it('treats a file that is not a zip as unreadable, not as over the limit', () => {
    let error: unknown;
    try {
      assertXlsxWithinBudget(Buffer.from('timestamp,glucose\n'));
    } catch (e) {
      error = e;
    }
    expect(error).toBeInstanceOf(Error);
    expect(error).not.toBeInstanceOf(ImportLimitError);
  });
});
