import { readFileSync } from 'node:fs';
import path from 'node:path';
import { afterAll, describe, expect, it } from 'vitest';
import {
  commitImport,
  ImportError,
  prepareImport,
  undoImport,
} from '@/lib/services/import-service';
import {
  prisma,
  createTestUser,
  deleteTestUser,
  toArrayBuffer,
  type SeededUser,
} from './test-helpers';

const createdUserIds: string[] = [];

afterAll(async () => {
  for (const id of createdUserIds) await deleteTestUser(id);
});

function readBytes(relPath: string): ArrayBuffer {
  const buf = readFileSync(path.join(process.cwd(), relPath));
  return toArrayBuffer(buf);
}

async function newUser(label: string): Promise<SeededUser> {
  const seeded = await createTestUser(label);
  createdUserIds.push(seeded.user.id);
  return seeded;
}

describe('end-to-end import: ml/data sample files', () => {
  it('imports ml/data/sample_logs.csv with correct sourcing, batch counts and provenance', async () => {
    const { user } = await newUser('import-logs');
    const bytes = readBytes('ml/data/sample_logs.csv');
    const prepared = await prepareImport({
      userId: user.id,
      filename: 'sample_logs.csv',
      mimeType: 'text/csv',
      bytes,
      timezone: 'UTC',
    });
    expect(prepared.connectorId).toBe('dialog-legacy');
    expect(prepared.summary.rowsImported).toBeGreaterThan(0);

    const { batchId, imported } = await commitImport({
      userId: user.id,
      filename: 'sample_logs.csv',
      mimeType: 'text/csv',
      byteSize: bytes.byteLength,
      prepared,
    });
    expect(imported).toBe(prepared.summary.rowsImported);

    const batch = await prisma.importBatch.findUniqueOrThrow({ where: { id: batchId } });
    expect(batch.status).toBe('COMPLETED');
    expect(batch.rowsImported).toBe(imported);
    expect(batch.rowsTotal).toBe(prepared.summary.rowsTotal);

    const rows = await prisma.glucoseReading.findMany({
      where: { userId: user.id, importBatchId: batchId },
    });
    expect(rows.length).toBeGreaterThan(0);
    for (const row of rows) {
      expect(row.source).toBe('IMPORT');
      expect(row.rawPayload).not.toBeNull();
    }

    const meds = await prisma.medicationEvent.findMany({
      where: { userId: user.id, source: 'IMPORT' },
    });
    expect(meds.length).toBeGreaterThan(0);
    const meals = await prisma.meal.findMany({ where: { userId: user.id, source: 'IMPORT' } });
    expect(meals.length).toBeGreaterThan(0);
  });

  it('imports ml/data/sample_glucose_data.csv (wide format) with correct sourcing', async () => {
    const { user } = await newUser('import-wide');
    const bytes = readBytes('ml/data/sample_glucose_data.csv');
    const prepared = await prepareImport({
      userId: user.id,
      filename: 'sample_glucose_data.csv',
      mimeType: 'text/csv',
      bytes,
      timezone: 'UTC',
    });
    expect(prepared.connectorId).toBe('dialog-legacy');

    const { batchId, imported } = await commitImport({
      userId: user.id,
      filename: 'sample_glucose_data.csv',
      mimeType: 'text/csv',
      byteSize: bytes.byteLength,
      prepared,
    });
    expect(imported).toBeGreaterThan(0);
    const glucoseRows = await prisma.glucoseReading.count({
      where: { userId: user.id, importBatchId: batchId, source: 'IMPORT' },
    });
    expect(glucoseRows).toBeGreaterThan(0);
  });
});

describe('re-importing the same file', () => {
  it('adds zero rows the second time and reports duplicates; total row count is unchanged', async () => {
    const { user } = await newUser('import-twice');
    const bytes = readBytes('ml/data/sample_logs.csv');

    const first = await prepareImport({
      userId: user.id,
      filename: 'sample_logs.csv',
      mimeType: 'text/csv',
      bytes,
      timezone: 'UTC',
    });
    const firstCommit = await commitImport({
      userId: user.id,
      filename: 'sample_logs.csv',
      mimeType: 'text/csv',
      byteSize: bytes.byteLength,
      prepared: first,
    });
    const countAfterFirst = await prisma.glucoseReading.count({ where: { userId: user.id } });
    expect(firstCommit.imported).toBeGreaterThan(0);

    const second = await prepareImport({
      userId: user.id,
      filename: 'sample_logs.csv',
      mimeType: 'text/csv',
      bytes,
      timezone: 'UTC',
    });
    expect(second.summary.rowsImported).toBe(0);
    expect(second.summary.rowsDuplicate).toBe(first.summary.rowsImported);

    const secondCommit = await commitImport({
      userId: user.id,
      filename: 'sample_logs.csv',
      mimeType: 'text/csv',
      byteSize: bytes.byteLength,
      prepared: second,
    });
    expect(secondCommit.imported).toBe(0);

    const countAfterSecond = await prisma.glucoseReading.count({ where: { userId: user.id } });
    expect(countAfterSecond).toBe(countAfterFirst);
  });
});

describe('malformed and all-invalid files', () => {
  it('fails gracefully (ImportError) for a corrupt/unrecognisable file, without writing anything', async () => {
    const { user } = await newUser('import-corrupt');
    const bytes = readBytes('tests/fixtures/integration/broken.json');
    await expect(
      prepareImport({
        userId: user.id,
        filename: 'broken.json',
        mimeType: 'application/json',
        bytes,
        timezone: 'UTC',
      }),
    ).rejects.toBeInstanceOf(ImportError);

    expect(await prisma.importBatch.count({ where: { userId: user.id } })).toBe(0);
    expect(await prisma.glucoseReading.count({ where: { userId: user.id } })).toBe(0);
  });

  it('rejects a file whose layout no connector recognises, without writing anything', async () => {
    const { user } = await newUser('import-unrecognised');
    const garbage = Buffer.from(
      'this is just some prose, not tabular health data at all\nmore prose here\n',
      'utf8',
    );
    const bytes = toArrayBuffer(garbage);
    await expect(
      prepareImport({
        userId: user.id,
        filename: 'notes.csv',
        mimeType: 'text/csv',
        bytes,
        timezone: 'UTC',
      }),
    ).rejects.toBeInstanceOf(ImportError);
    expect(await prisma.importBatch.count({ where: { userId: user.id } })).toBe(0);
  });

  it('a file where every row is invalid parses with zero fresh records and every row recorded as an issue — never a partial silent write', async () => {
    const { user } = await newUser('import-all-invalid');
    const bytes = readBytes('tests/fixtures/integration/all-invalid.csv');
    const prepared = await prepareImport({
      userId: user.id,
      filename: 'all-invalid.csv',
      mimeType: 'text/csv',
      bytes,
      timezone: 'UTC',
    });
    expect(prepared.fresh).toHaveLength(0);
    expect(prepared.summary.rowsImported).toBe(0);
    expect(prepared.summary.rowsRejected).toBe(prepared.summary.rowsTotal);
    expect(prepared.summary.issueGroups.length).toBeGreaterThan(0);

    const { batchId, imported } = await commitImport({
      userId: user.id,
      filename: 'all-invalid.csv',
      mimeType: 'text/csv',
      byteSize: bytes.byteLength,
      prepared,
    });
    expect(imported).toBe(0);
    expect(await prisma.glucoseReading.count({ where: { userId: user.id } })).toBe(0);
    const batch = await prisma.importBatch.findUniqueOrThrow({ where: { id: batchId } });
    expect(batch.status).toBe('COMPLETED');
    expect(batch.rowsImported).toBe(0);
    const issues = await prisma.importIssue.findMany({ where: { batchId } });
    expect(issues.length).toBeGreaterThan(0);
  });
});

describe('mmol/L file is converted to mg/dL on storage', () => {
  it('stores canonical mg/dL values converted from the mmol/L source file', async () => {
    const { user } = await newUser('import-mmol');
    const bytes = readBytes('tests/fixtures/integration/mmol-readings.csv');
    const prepared = await prepareImport({
      userId: user.id,
      filename: 'mmol-readings.csv',
      mimeType: 'text/csv',
      bytes,
      timezone: 'UTC',
    });
    expect(prepared.summary.detectedUnit).toBe('MMOLL');
    expect(prepared.fresh).toHaveLength(3);

    await commitImport({
      userId: user.id,
      filename: 'mmol-readings.csv',
      mimeType: 'text/csv',
      byteSize: bytes.byteLength,
      prepared,
    });

    const rows = await prisma.glucoseReading.findMany({
      where: { userId: user.id },
      orderBy: { takenAt: 'asc' },
    });
    expect(rows).toHaveLength(3);
    // 5.5 mmol/L * 18.0182 ≈ 99.1 mg/dL — stored value must be the converted mg/dL, not the raw 5.5.
    expect(rows[0]?.valueMgdl).toBeGreaterThan(90);
    expect(rows[0]?.valueMgdl).toBeLessThan(110);
    expect(rows[0]?.valueMgdl).toBeCloseTo(5.5 * 18.0182, 1);
    expect(rows[1]?.valueMgdl).toBeCloseTo(7.2 * 18.0182, 1);
    expect(rows[2]?.valueMgdl).toBeCloseTo(9.8 * 18.0182, 1);
  });
});

describe('undoImport', () => {
  it("removes exactly the batch's rows and nothing else, including nothing belonging to another user", async () => {
    const { user: owner } = await newUser('undo-owner');
    const { user: other } = await newUser('undo-other');

    // A pre-existing manual row for the owner that must survive the undo.
    await prisma.glucoseReading.create({
      data: { userId: owner.id, takenAt: new Date(), valueMgdl: 111, dedupeKey: 'undo-manual' },
    });
    // Another user's data that must never be touched by this undo.
    await prisma.glucoseReading.create({
      data: {
        userId: other.id,
        takenAt: new Date(),
        valueMgdl: 222,
        dedupeKey: 'undo-other-manual',
      },
    });

    // A pure-glucose fixture (no meals/medications) so "imported" and "removed"
    // are directly comparable — see the mixed-batch test below for the gap
    // where that is NOT true.
    const bytes = readBytes('tests/fixtures/integration/mmol-readings.csv');
    const prepared = await prepareImport({
      userId: owner.id,
      filename: 'mmol-readings.csv',
      mimeType: 'text/csv',
      bytes,
      timezone: 'UTC',
    });
    const { batchId, imported } = await commitImport({
      userId: owner.id,
      filename: 'mmol-readings.csv',
      mimeType: 'text/csv',
      byteSize: bytes.byteLength,
      prepared,
    });
    expect(imported).toBe(3);

    const removed = await undoImport(owner.id, batchId);
    expect(removed).toBe(imported);

    expect(await prisma.importBatch.findUnique({ where: { id: batchId } })).toBeNull();
    expect(
      await prisma.glucoseReading.count({ where: { userId: owner.id, importBatchId: batchId } }),
    ).toBe(0);

    // The pre-existing manual row for the owner survives.
    expect(await prisma.glucoseReading.count({ where: { userId: owner.id, valueMgdl: 111 } })).toBe(
      1,
    );
    // The other user's row is completely untouched.
    expect(await prisma.glucoseReading.count({ where: { userId: other.id, valueMgdl: 222 } })).toBe(
      1,
    );
  });

  it('undoImport for a batch owned by someone else does nothing and returns 0', async () => {
    const { user: owner } = await newUser('undo-owner-2');
    const { user: attacker } = await newUser('undo-attacker');

    const bytes = readBytes('tests/fixtures/integration/mmol-readings.csv');
    const prepared = await prepareImport({
      userId: owner.id,
      filename: 'mmol-readings.csv',
      mimeType: 'text/csv',
      bytes,
      timezone: 'UTC',
    });
    const { batchId, imported } = await commitImport({
      userId: owner.id,
      filename: 'mmol-readings.csv',
      mimeType: 'text/csv',
      byteSize: bytes.byteLength,
      prepared,
    });
    expect(imported).toBe(3);

    const removedByAttacker = await undoImport(attacker.id, batchId);
    expect(removedByAttacker).toBe(0);
    expect(await prisma.importBatch.findUnique({ where: { id: batchId } })).not.toBeNull();
    expect(
      await prisma.glucoseReading.count({ where: { userId: owner.id, importBatchId: batchId } }),
    ).toBe(imported);
  });

  // KNOWN BUG (reported, not fixed — see final summary): undoImport's own doc
  // comment says it removes "an entire import batch and everything it
  // created", and commitImport's `imported` return value counts every kind of
  // row the batch wrote (glucose, meals, AND medications/exercise/sleep/
  // weight/notes). But undoImport only deletes GlucoseReading and Meal rows
  // (the only two models with an importBatchId column) — medication rows
  // created by the same import are silently left behind once the
  // ImportBatch row is deleted, with no way to trace them back to a batch
  // afterward. A user who imports sample_logs.csv (glucose + meal + med
  // rows) and clicks "undo" is told their import was undone, but their
  // medication history still contains rows from that file.
  it('undoImport removes every record type the batch created, not only glucose and meals', async () => {
    const { user: owner } = await newUser('undo-mixed-batch');

    const bytes = readBytes('ml/data/sample_logs.csv');
    const prepared = await prepareImport({
      userId: owner.id,
      filename: 'sample_logs.csv',
      mimeType: 'text/csv',
      bytes,
      timezone: 'UTC',
    });
    const { batchId, imported } = await commitImport({
      userId: owner.id,
      filename: 'sample_logs.csv',
      mimeType: 'text/csv',
      byteSize: bytes.byteLength,
      prepared,
    });

    // sample_logs.csv mixes glucose, meal and medication rows.
    const medsFromThisImport = await prisma.medicationEvent.count({
      where: { userId: owner.id, importBatchId: batchId },
    });
    expect(medsFromThisImport).toBeGreaterThan(0);

    const removed = await undoImport(owner.id, batchId);

    expect(await prisma.importBatch.findUnique({ where: { id: batchId } })).toBeNull();
    expect(await prisma.glucoseReading.count({ where: { userId: owner.id } })).toBe(0);
    expect(await prisma.meal.count({ where: { userId: owner.id } })).toBe(0);
    expect(await prisma.medicationEvent.count({ where: { userId: owner.id } })).toBe(0);
    // Everything the import wrote is accounted for in the returned count.
    expect(removed).toBe(imported);
  });
});
