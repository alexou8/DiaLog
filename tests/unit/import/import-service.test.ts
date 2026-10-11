/**
 * prepareImport used to load every dedupe key the user had ever stored (about
 * 200k rows for two years of CGM) on each analyse and commit. It now asks
 * only for the keys in the file, chunked. Prisma is mocked: this pins the
 * query shape and the user-facing errors, not the database.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest';

const findMany = vi.fn();

vi.mock('@/lib/db/prisma', () => {
  const delegate = { findMany: (...args: unknown[]) => findMany(...args) };
  return {
    prisma: {
      glucoseReading: delegate,
      meal: delegate,
      medicationEvent: delegate,
      exerciseSession: delegate,
      sleepSession: delegate,
      weightMeasurement: delegate,
      noteEntry: delegate,
    },
  };
});

const { prepareImport, ImportError } = await import('@/lib/services/import-service');
const { keyRecord } = await import('@/lib/import/dedupe');
const { MAX_IMPORT_BYTES } = await import('@/lib/import/limits');

function csvBytes(text: string): ArrayBuffer {
  const buf = Buffer.from(text, 'utf8');
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer;
}

function glucoseCsv(rows: number): string {
  const lines = ['timestamp,glucose'];
  const start = Date.UTC(2026, 0, 1);
  for (let i = 0; i < rows; i++) {
    lines.push(`${new Date(start + i * 5 * 60_000).toISOString()},${100 + (i % 50)}`);
  }
  return lines.join('\n');
}

const base = { userId: 'u1', filename: 'readings.csv', mimeType: 'text/csv', timezone: 'UTC' };

beforeEach(() => {
  findMany.mockReset();
  findMany.mockResolvedValue([]);
});

describe('prepareImport dedupe lookup', () => {
  it('queries only the keys present in the file, in chunks of at most 1000', async () => {
    const prepared = await prepareImport({ ...base, bytes: csvBytes(glucoseCsv(2_500)) });
    expect(prepared.summary.rowsImported).toBe(2_500);
    expect(findMany).toHaveBeenCalledTimes(3);
    const sizes = findMany.mock.calls.map(
      ([args]) => (args as { where: { dedupeKey: { in: string[] } } }).where.dedupeKey.in.length,
    );
    expect(sizes).toEqual([1_000, 1_000, 500]);
    for (const [args] of findMany.mock.calls) {
      expect((args as { where: { userId: string } }).where.userId).toBe('u1');
    }
  });

  it('counts rows whose keys already exist as duplicates', async () => {
    const first = await prepareImport({ ...base, bytes: csvBytes(glucoseCsv(3)) });
    const storedKey = first.fresh[0]!.dedupeKey;
    expect(storedKey).toBe(keyRecord(first.fresh[0]!.record).dedupeKey);

    findMany.mockReset();
    findMany.mockResolvedValue([{ dedupeKey: storedKey }]);
    const second = await prepareImport({ ...base, bytes: csvBytes(glucoseCsv(3)) });
    expect(second.summary.rowsDuplicate).toBe(1);
    expect(second.summary.rowsImported).toBe(2);
  });
});

describe('prepareImport errors', () => {
  it('refuses an over-limit file with the size in the message', async () => {
    const error = await prepareImport({
      ...base,
      bytes: new ArrayBuffer(MAX_IMPORT_BYTES + 1),
    }).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ImportError);
    expect((error as Error).message).toMatch(/20 MB/);
  });

  it('explains an XML file refused for declaring entities', async () => {
    const error = await prepareImport({
      ...base,
      filename: 'export.xml',
      mimeType: 'application/xml',
      bytes: csvBytes('<!DOCTYPE r [<!ENTITY e "x">]><r>&e;</r>'),
    }).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ImportError);
    expect((error as Error).message).toMatch(/entities/);
  });

  it('refuses an unrecognised time zone instead of importing at UTC', async () => {
    await expect(
      prepareImport({ ...base, timezone: 'Mars/Olympus', bytes: csvBytes(glucoseCsv(1)) }),
    ).rejects.toThrow(/time zone/);
  });
});
