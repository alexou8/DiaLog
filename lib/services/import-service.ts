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
import { dedupeRecords, type KeyedRecord } from '@/lib/import/dedupe';
import { detectConnector } from '@/lib/import/connectors/registry';
import { parseFile, MAX_FILE_BYTES } from '@/lib/import/parse';
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
  if (params.bytes.byteLength > MAX_FILE_BYTES) {
    throw new ImportError(
      'That file is larger than DiaLog can process. Please split it into smaller exports.',
    );
  }

  let parsedFile;
  try {
    parsedFile = await parseFile(params.filename, params.mimeType, Buffer.from(params.bytes));
  } catch {
    throw new ImportError(
      'DiaLog could not read that file. It may be corrupted, or it may be a format DiaLog does not understand yet.',
    );
  }

  const detection = detectConnector(parsedFile);
  if (!detection) {
    throw new ImportError(
      'DiaLog could not recognise the layout of that file. The supported formats are listed on this page — a plain CSV with a date column and a glucose column always works.',
    );
  }

  const parseResult = await detection.connector.parse(parsedFile, { timezone: params.timezone });

  const existing = await existingKeysFor(params.userId, parseResult.records);
  const dedupe = dedupeRecords(parseResult.records, existing);

  return {
    connectorId: detection.connector.id,
    connectorName: detection.connector.name,
    summary: buildImportSummary(parseResult, dedupe),
    fresh: dedupe.fresh,
    preview: dedupe.fresh.slice(0, 5).map((entry) => entry.record),
  };
}

/**
 * Look up which of these records the user already has. Only the dedupe keys are
 * fetched, and only for the record kinds present in the file.
 */
async function existingKeysFor(
  userId: string,
  records: readonly NormalizedRecord[],
): Promise<Set<string>> {
  const kinds = new Set(records.map((record) => record.kind));
  const keys = new Set<string>();
  const select = { dedupeKey: true } as const;

  const lookups: Promise<{ dedupeKey: string }[]>[] = [];
  if (kinds.has('glucose'))
    lookups.push(prisma.glucoseReading.findMany({ where: { userId }, select }));
  if (kinds.has('meal')) lookups.push(prisma.meal.findMany({ where: { userId }, select }));
  if (kinds.has('medication'))
    lookups.push(prisma.medicationEvent.findMany({ where: { userId }, select }));
  if (kinds.has('exercise'))
    lookups.push(prisma.exerciseSession.findMany({ where: { userId }, select }));
  if (kinds.has('sleep')) lookups.push(prisma.sleepSession.findMany({ where: { userId }, select }));
  if (kinds.has('weight'))
    lookups.push(prisma.weightMeasurement.findMany({ where: { userId }, select }));
  if (kinds.has('note')) lookups.push(prisma.noteEntry.findMany({ where: { userId }, select }));

  for (const rows of await Promise.all(lookups)) {
    for (const row of rows) keys.add(row.dedupeKey);
  }
  return keys;
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
