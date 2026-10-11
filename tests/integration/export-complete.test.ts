/**
 * The data export is complete and safe to open in a spreadsheet.
 *
 * It used to stop at 20,000 rows per record type — about ten weeks of CGM
 * readings — so "download all my data" quietly returned a fraction of a long
 * history. It now pages through everything by (takenAt, id). These tests use
 * a tiny page size to cross many page boundaries cheaply, put ties on
 * `takenAt` right across those boundaries (the case an offset- or
 * timestamp-only cursor gets wrong), and run once at the real page size.
 */
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import {
  EXPORT_PAGE_SIZE,
  EXPORT_SCHEMA_VERSION,
  buildCsvExport,
  buildJsonExport,
} from '@/lib/services/export-service';
import { prisma, createTestUser, deleteTestUser, type SeededUser } from './test-helpers';

let owner: SeededUser;
let other: SeededUser;

const READINGS = 23;
const PAGE = 4;

beforeAll(async () => {
  owner = await createTestUser('export-owner');
  other = await createTestUser('export-other');
  // Groups of three readings share an instant, so most page edges (every
  // fourth row) land inside a tie.
  await prisma.glucoseReading.createMany({
    data: Array.from({ length: READINGS }, (_, i) => ({
      userId: owner.user.id,
      takenAt: new Date(Date.UTC(2026, 0, 1, Math.floor(i / 3))),
      valueMgdl: 90 + i,
      context: 'RANDOM' as const,
      dedupeKey: `export-${i}`,
    })),
  });
  await prisma.glucoseReading.create({
    data: {
      userId: other.user.id,
      takenAt: new Date(Date.UTC(2026, 0, 1, 2)),
      valueMgdl: 999,
      dedupeKey: 'export-other',
    },
  });
  await prisma.noteEntry.createMany({
    data: [
      {
        userId: owner.user.id,
        takenAt: new Date('2026-01-02T08:00:00Z'),
        text: '=HYPERLINK("https://evil.example","Open")',
        dedupeKey: 'export-note-1',
      },
      {
        userId: owner.user.id,
        takenAt: new Date('2026-01-02T09:00:00Z'),
        text: '-2 units correction, felt fine',
        dedupeKey: 'export-note-2',
      },
    ],
  });
  await prisma.weightMeasurement.create({
    data: {
      userId: owner.user.id,
      takenAt: new Date('2026-01-03T08:00:00Z'),
      weightKg: 71.5,
      note: '@home scale',
      dedupeKey: 'export-weight-1',
    },
  });
});

afterAll(async () => {
  await deleteTestUser(owner.user.id);
  await deleteTestUser(other.user.id);
});

async function ownerIdsInOrder(): Promise<string[]> {
  const rows = await prisma.glucoseReading.findMany({
    where: { userId: owner.user.id },
    orderBy: [{ takenAt: 'asc' }, { id: 'asc' }],
    select: { id: true },
  });
  return rows.map((r) => r.id);
}

describe('export completeness across pages', () => {
  it('CSV contains every reading exactly once, in order, and only the owner’s', async () => {
    const result = await buildCsvExport(owner.user.id, 'glucose', { pageSize: PAGE });
    const lines = result.csv.trimEnd().split('\r\n');
    expect(lines[0]).toBe('id,taken_at,value_mgdl,value_mmoll,context,note,source');
    const ids = lines.slice(1).map((line) => line.split(',')[0]);
    expect(ids).toEqual(await ownerIdsInOrder());
    expect(result.rowCount).toBe(READINGS);
    expect(result.truncated).toBe(false);
    expect(result.csv).not.toContain(',999,');
  });

  it('JSON contains every reading exactly once, reports the count, and is valid JSON', async () => {
    const data = await buildJsonExport(owner.user.id, { pageSize: PAGE });
    expect(data.schemaVersion).toBe(EXPORT_SCHEMA_VERSION);
    expect(data.rowLimit).toBeNull();
    const ids = (data.records.glucose as { id: string }[]).map((r) => r.id);
    expect(ids).toEqual(await ownerIdsInOrder());
    expect(data.meta.glucose).toEqual({ truncated: false, count: READINGS });
    expect(data.meta.note).toEqual({ truncated: false, count: 2 });
    expect(data.meta.meal).toEqual({ truncated: false, count: 0 });
    expect(data.records.meals).toEqual([]);
  });

  it('crosses a real page boundary at the production page size', async () => {
    const user = await createTestUser('export-large');
    try {
      const total = EXPORT_PAGE_SIZE + 7;
      await prisma.glucoseReading.createMany({
        data: Array.from({ length: total }, (_, i) => ({
          userId: user.user.id,
          takenAt: new Date(Date.UTC(2025, 0, 1, 0, 5 * i)),
          valueMgdl: 100,
          dedupeKey: `export-large-${i}`,
        })),
      });
      const csv = await buildCsvExport(user.user.id, 'glucose');
      expect(csv.rowCount).toBe(total);
      const ids = csv.csv
        .trimEnd()
        .split('\r\n')
        .slice(1)
        .map((line) => line.split(',')[0]);
      expect(new Set(ids).size).toBe(total);

      const json = await buildJsonExport(user.user.id);
      expect(json.records.glucose).toHaveLength(total);
      expect(json.meta.glucose?.count).toBe(total);
    } finally {
      await deleteTestUser(user.user.id);
    }
  });
});

describe('CSV formula neutralisation', () => {
  it('prefixes text cells that a spreadsheet would run as a formula', async () => {
    const notes = await buildCsvExport(owner.user.id, 'note');
    expect(notes.csv).toContain(`"'=HYPERLINK(""https://evil.example"",""Open"")"`);
    expect(notes.csv).toContain(`'-2 units correction`);
    const weight = await buildCsvExport(owner.user.id, 'weight');
    expect(weight.csv).toContain(`,71.5,'@home scale,`);
  });

  it('leaves numeric columns as numbers', async () => {
    const glucose = await buildCsvExport(owner.user.id, 'glucose');
    const firstRow = glucose.csv.split('\r\n')[1]!.split(',');
    expect(firstRow[2]).toBe('90');
    expect(firstRow[2]!.startsWith("'")).toBe(false);
  });

  it('does not alter the JSON export, which is not a spreadsheet format', async () => {
    const data = await buildJsonExport(owner.user.id);
    const texts = (data.records.notes as { text: string }[]).map((n) => n.text);
    expect(texts).toContain('=HYPERLINK("https://evil.example","Open")');
  });
});
