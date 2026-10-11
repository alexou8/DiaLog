/**
 * Import orchestration: file → connector → normalised records → database.
 *
 * The import is a two-stage flow on purpose. Stage one parses and reports, and
 * saves nothing; stage two writes only after the user has seen what will
 * happen. Nothing about a user's file is ever altered silently — a row that
 * cannot be trusted becomes a visible issue rather than a corrected value.
 */
import type { DataSource, Prisma } from '@prisma/client';
import { prisma } from '@/lib/db/prisma';
import { isValidTimeZone } from '@/lib/domain/time';
import { dedupeKeyedRecords, keyRecord, type KeyedRecord } from '@/lib/import/dedupe';
import { detectConnector } from '@/lib/import/connectors/registry';
import { ImportLimitError, MAX_IMPORT_BYTES, fileTooLargeMessage } from '@/lib/import/limits';
import { enforceStorageConstraints } from '@/lib/import/normalize';
import { parseFile } from '@/lib/import/parse';
import { buildImportSummary, type ImportSummary } from '@/lib/import/summary';
import type { NormalizedRecord } from '@/lib/import/types';

export interface PreparedImport {
  connectorId: string;
  connectorName: string;
  summary: ImportSummary;
  fresh: KeyedRecord[];
  /** A handful of records shown to the user as a preview before committing. */
  preview: NormalizedRecord[];
}

export class ImportError extends Error {}

/** Parse and analyse an uploaded file without writing anything. */
export async function prepareImport(params: {
  userId: string;
  filename: string;
  mimeType: string;
  bytes: ArrayBuffer;
  timezone: string;
}): Promise<PreparedImport> {
  if (params.bytes.byteLength === 0) {
    throw new ImportError(
      'That file is empty. Please check you exported the right file and try again.',
    );
  }
  if (params.bytes.byteLength > MAX_IMPORT_BYTES) {
    throw new ImportError(fileTooLargeMessage());
  }
  // Checked here rather than per row: an unrecognised zone used to be read
  // silently as UTC, shifting every imported reading by the zone's offset.
  if (!isValidTimeZone(params.timezone)) {
    throw new ImportError(
      'Your time zone setting is not recognised, so imported times could not be placed correctly. Please choose your time zone again in Settings, then retry the import.',
    );
  }

  let parsedFile;
  try {
    parsedFile = await parseFile(params.filename, params.mimeType, Buffer.from(params.bytes));
  } catch (error) {
    throw importErrorFor(error);
  }

  const detection = detectConnector(parsedFile);
  if (!detection) {
    throw new ImportError(
      'DiaLog could not recognise the layout of that file. The supported formats are listed on this page. A plain CSV with a date column and a glucose column always works.',
    );
  }

  // The backstop runs on every connector's output, so no connector can send a
  // value Prisma or Postgres would refuse (which fails the whole commit).
  const parseResult = enforceStorageConstraints(
    await detection.connector.parse(parsedFile, { timezone: params.timezone }),
  );

  const keyed = parseResult.records.map(keyRecord);
  const existing = await existingKeysFor(params.userId, keyed);
  const dedupe = dedupeKeyedRecords(keyed, existing);

  return {
    connectorId: detection.connector.id,
    connectorName: detection.connector.name,
    summary: buildImportSummary(parseResult, dedupe),
    fresh: dedupe.fresh,
    preview: dedupe.fresh.slice(0, 5).map((entry) => entry.record),
  };
}

/** Maps a parse failure to the message the user sees. Never echoes file contents. */
function importErrorFor(error: unknown): ImportError {
  if (error instanceof ImportLimitError) {
    switch (error.reason) {
      case 'FILE_TOO_LARGE':
        return new ImportError(fileTooLargeMessage());
      case 'WORKBOOK_TOO_LARGE':
        // The limit's own message is fixed, user-facing copy.
        return new ImportError(error.message);
      case 'UNSAFE_XML':
        return new ImportError(
          'That XML file declares entities or an external DTD, which DiaLog does not accept because they can be used to exhaust memory. Exports from Apple Health and other supported apps do not need them.',
        );
    }
  }
  return new ImportError(
    'DiaLog could not read that file. It may be corrupted, or it may be a format DiaLog does not understand yet.',
  );
}

/** Keys per `IN (...)` query; well under Postgres' bind-parameter limit. */
const DEDUPE_LOOKUP_CHUNK = 1_000;

/**
 * Look up which of these records the user already has.
 *
 * Only the keys present in the file are queried (`dedupeKey IN (...)`, in
 * chunks, against the `@@unique([userId, dedupeKey])` index). This used to
 * fetch every key the user had ever stored, per kind — about 15 MB and 200k
 * rows for two years of CGM data, on every analyse and again on commit.
 * `createMany({ skipDuplicates })` at commit remains the backstop for a key
 * written concurrently after this lookup.
 */
async function existingKeysFor(
  userId: string,
  keyed: readonly KeyedRecord[],
): Promise<Set<string>> {
  const keysByKind = new Map<NormalizedRecord['kind'], Set<string>>();
  for (const { record, dedupeKey } of keyed) {
    let set = keysByKind.get(record.kind);
    if (!set) {
      set = new Set();
      keysByKind.set(record.kind, set);
    }
    set.add(dedupeKey);
  }

  const existing = new Set<string>();
  for (const [kind, keys] of keysByKind) {
    const all = [...keys];
    for (let i = 0; i < all.length; i += DEDUPE_LOOKUP_CHUNK) {
      const rows = await findExistingKeys(kind, userId, all.slice(i, i + DEDUPE_LOOKUP_CHUNK));
      for (const row of rows) existing.add(row.dedupeKey);
    }
  }
  return existing;
}

function findExistingKeys(
  kind: NormalizedRecord['kind'],
  userId: string,
  keys: string[],
): Promise<{ dedupeKey: string }[]> {
  const args = { where: { userId, dedupeKey: { in: keys } }, select: { dedupeKey: true } } as const;
  switch (kind) {
    case 'glucose':
      return prisma.glucoseReading.findMany(args);
    case 'meal':
      return prisma.meal.findMany(args);
    case 'medication':
      return prisma.medicationEvent.findMany(args);
    case 'exercise':
      return prisma.exerciseSession.findMany(args);
    case 'sleep':
      return prisma.sleepSession.findMany(args);
    case 'weight':
      return prisma.weightMeasurement.findMany(args);
    case 'note':
      return prisma.noteEntry.findMany(args);
  }
}

/** Write a prepared import, recording the batch so it can be traced or undone. */
export async function commitImport(params: {
  userId: string;
  filename: string;
  mimeType: string;
  byteSize: number;
  prepared: PreparedImport;
}): Promise<{ batchId: string; imported: number }> {
  const { prepared } = params;
  const source: DataSource = 'IMPORT';

  const batch = await prisma.importBatch.create({
    data: {
      userId: params.userId,
      connectorId: prepared.connectorId,
      connectorName: prepared.connectorName,
      filename: params.filename,
      mimeType: params.mimeType,
      byteSize: params.byteSize,
      status: 'PENDING',
      rowsTotal: prepared.summary.rowsTotal,
      rowsDuplicate: prepared.summary.rowsDuplicate,
      rowsRejected: prepared.summary.rowsRejected,
      issues: {
        create: prepared.summary.issueGroups.flatMap((group) =>
          group.examples.slice(0, 20).map((issue) => ({
            rowNumber: issue.rowNumber,
            code: issue.code,
            message: issue.message,
            rawRow: issue.rawRow ?? null,
          })),
        ),
      },
    },
  });

  const glucose: Prisma.GlucoseReadingCreateManyInput[] = [];
  const meals: Prisma.MealCreateManyInput[] = [];
  const medications: Prisma.MedicationEventCreateManyInput[] = [];
  const exercise: Prisma.ExerciseSessionCreateManyInput[] = [];
  const sleep: Prisma.SleepSessionCreateManyInput[] = [];
  const weight: Prisma.WeightMeasurementCreateManyInput[] = [];
  const notes: Prisma.NoteEntryCreateManyInput[] = [];

  for (const { record, dedupeKey } of prepared.fresh) {
    const common = { userId: params.userId, takenAt: record.takenAt, source, dedupeKey };
    switch (record.kind) {
      case 'glucose':
        glucose.push({
          ...common,
          valueMgdl: record.valueMgdl,
          context: record.context,
          note: record.note ?? null,
          externalId: record.externalId ?? null,
          importBatchId: batch.id,
          rawPayload: record.raw as Prisma.InputJsonValue,
        });
        break;
      case 'meal':
        meals.push({
          ...common,
          description: record.description,
          mealType: record.mealType,
          carbsG: record.carbsG ?? null,
          proteinG: record.proteinG ?? null,
          fatG: record.fatG ?? null,
          fiberG: record.fiberG ?? null,
          calories: record.calories ?? null,
          estimateSource: 'IMPORTED',
          importBatchId: batch.id,
          rawPayload: record.raw as Prisma.InputJsonValue,
        });
        break;
      case 'medication':
        medications.push({
          ...common,
          name: record.name,
          dose: record.dose ?? null,
          importBatchId: batch.id,
          rawPayload: record.raw as Prisma.InputJsonValue,
        });
        break;
      case 'exercise':
        exercise.push({
          ...common,
          activity: record.activity,
          durationMin: record.durationMin,
          intensity: record.intensity ?? 'MODERATE',
          endedAt: new Date(record.takenAt.getTime() + record.durationMin * 60_000),
          distanceKm: record.distanceKm ?? null,
          steps: record.steps ?? null,
          importBatchId: batch.id,
          rawPayload: record.raw as Prisma.InputJsonValue,
        });
        break;
      case 'sleep':
        sleep.push({
          ...common,
          endedAt: record.endedAt,
          durationMin: record.durationMin,
          quality: record.quality ?? null,
          importBatchId: batch.id,
          rawPayload: record.raw as Prisma.InputJsonValue,
        });
        break;
      case 'weight':
        weight.push({
          ...common,
          weightKg: record.weightKg,
          importBatchId: batch.id,
          rawPayload: record.raw as Prisma.InputJsonValue,
        });
        break;
      case 'note':
        notes.push({
          ...common,
          text: record.text,
          importBatchId: batch.id,
          rawPayload: record.raw as Prisma.InputJsonValue,
        });
        break;
    }
  }

  // `skipDuplicates` is belt-and-braces: the dedupe pass above should already
  // have removed anything the user has, but a concurrent import must not fail
  // the whole batch.
  const results = await prisma.$transaction([
    prisma.glucoseReading.createMany({ data: glucose, skipDuplicates: true }),
    prisma.meal.createMany({ data: meals, skipDuplicates: true }),
    prisma.medicationEvent.createMany({ data: medications, skipDuplicates: true }),
    prisma.exerciseSession.createMany({ data: exercise, skipDuplicates: true }),
    prisma.sleepSession.createMany({ data: sleep, skipDuplicates: true }),
    prisma.weightMeasurement.createMany({ data: weight, skipDuplicates: true }),
    prisma.noteEntry.createMany({ data: notes, skipDuplicates: true }),
  ]);

  const imported = results.reduce((total, result) => total + result.count, 0);

  await prisma.importBatch.update({
    where: { id: batch.id },
    data: { status: 'COMPLETED', rowsImported: imported, finishedAt: new Date() },
  });

  return { batchId: batch.id, imported };
}

/**
 * Remove an entire import batch and every record it created.
 *
 * Each importable record type carries its `importBatchId`, so this covers all
 * of them — a partial undo that silently left rows behind would contradict
 * what the interface promises the user.
 */
export async function undoImport(userId: string, batchId: string): Promise<number> {
  const batch = await prisma.importBatch.findFirst({
    where: { id: batchId, userId },
    select: { id: true },
  });
  if (!batch) return 0;

  const where = { userId, importBatchId: batch.id };
  const results = await prisma.$transaction([
    prisma.glucoseReading.deleteMany({ where }),
    prisma.meal.deleteMany({ where }),
    prisma.medicationEvent.deleteMany({ where }),
    prisma.exerciseSession.deleteMany({ where }),
    prisma.sleepSession.deleteMany({ where }),
    prisma.weightMeasurement.deleteMany({ where }),
    prisma.noteEntry.deleteMany({ where }),
  ]);
  await prisma.importBatch.delete({ where: { id: batch.id } });
  return results.reduce((total, result) => total + result.count, 0);
}
