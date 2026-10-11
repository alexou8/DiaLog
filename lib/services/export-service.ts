/**
 * Builds a user's full data export.
 *
 * Every query here is scoped by `userId` — there is no code path in this
 * file that can read another account's data. Passwords and other internal
 * secrets are never touched; ids are kept on records (so a user can
 * cross-reference an import batch or a device) but dropped where they carry
 * no meaning outside DiaLog (e.g. the Profile row's own id).
 *
 * Complete, not capped. This used to stop at 20,000 rows per record type,
 * which a CGM user passes in about ten weeks (one reading every five minutes),
 * so "download all my data" quietly returned a fraction of it. Records are now
 * read in keyset pages of EXPORT_PAGE_SIZE ordered by (takenAt, id) and written
 * out as they arrive (`streamJsonExport` / `streamCsvExport`), so memory stays
 * at one page whatever the history length and there is no ceiling to hit.
 * Nothing is ever dropped silently: if a page fails, the stream errors and the
 * download fails visibly rather than ending early with a valid-looking file.
 *
 * Not a point-in-time snapshot: each page is its own query. A record written
 * while an export is running may or may not appear in it; keyset paging
 * guarantees no row present for the whole export is skipped or repeated.
 */
import type { Profile } from '@prisma/client';
import { prisma } from '@/lib/db/prisma';
import { fromMgdl } from '@/lib/domain/units';

/** Rows read per query. Bounds memory; has no effect on what is exported. */
export const EXPORT_PAGE_SIZE = 2_000;

/**
 * 2: `rowLimit` is now `null` (no cap) and `meta.*.truncated` is always false.
 * The record shapes are unchanged from version 1.
 */
export const EXPORT_SCHEMA_VERSION = 2;

export type ExportRecordType =
  | 'glucose'
  | 'meal'
  | 'exercise'
  | 'sleep'
  | 'medication'
  | 'weight'
  | 'bloodPressure'
  | 'hydration'
  | 'symptom'
  | 'mood'
  | 'note';

export const EXPORT_RECORD_TYPES: readonly ExportRecordType[] = [
  'glucose',
  'meal',
  'exercise',
  'sleep',
  'medication',
  'weight',
  'bloodPressure',
  'hydration',
  'symptom',
  'mood',
  'note',
];

export interface ExportMeta {
  /** Always false since schema version 2; kept so version-1 readers still parse. */
  truncated: boolean;
  count: number;
}

export interface JsonExport {
  exportedAt: string;
  schemaVersion: number;
  /** Always null since schema version 2: there is no per-type cap. */
  rowLimit: null;
  profile: {
    displayName: string | null;
    glucoseUnit: string;
    locale: string;
    timezone: string;
    condition: string;
    targetLowMgdl: number;
    targetHighMgdl: number;
    targetLowMmoll: number;
    targetHighMmoll: number;
    goals: string[];
    detailLevel: string;
    createdAt: string;
  };
  records: {
    glucose: unknown[];
    meals: unknown[];
    exercise: unknown[];
    sleep: unknown[];
    medications: unknown[];
    weight: unknown[];
    bloodPressure: unknown[];
    hydration: unknown[];
    symptoms: unknown[];
    moods: unknown[];
    notes: unknown[];
  };
  devices: unknown[];
  importBatches: unknown[];
  meta: Partial<Record<ExportRecordType, ExportMeta>>;
}

export interface ExportOptions {
  /** Override the page size. Tests use a tiny one to cross many page boundaries cheaply. */
  pageSize?: number;
}

function round(value: number, digits: number): number {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

// ----------------------------------------------------------- keyset paging

interface Position {
  takenAt: Date;
  id: string;
}

/** Strictly after `position` in (takenAt asc, id asc) order. */
function after(position: Position | null) {
  return position
    ? {
        OR: [
          { takenAt: { gt: position.takenAt } },
          { takenAt: position.takenAt, id: { gt: position.id } },
        ],
      }
    : {};
}

const ORDER = [{ takenAt: 'asc' as const }, { id: 'asc' as const }];

/** One exported row, in both shapes, plus where it sits in the keyset order. */
interface ExportRow {
  position: Position;
  json: Record<string, unknown>;
  /** Cells in header order. Text cells are already formula-neutralised. */
  csv: unknown[];
}

interface RecordSource {
  /** File stem for the CSV download. */
  file: string;
  /** Key under `records` in the JSON export. */
  jsonKey: keyof JsonExport['records'];
  header: string[];
  page(userId: string, from: Position | null, take: number): Promise<ExportRow[]>;
}

/**
 * CSV / formula injection. A cell whose text starts with `=`, `+`, `-`, `@`,
 * tab or carriage return is treated as a formula by Excel, Sheets and
 * LibreOffice, so a note such as `=HYPERLINK("https://evil.example","Open")`
 * — typed by the user or, more likely, carried in from an imported third-party
 * file — would run in the spreadsheet of whoever opens the export (perhaps a
 * clinician). Prefixing a single quote makes the application show it as text.
 *
 * Applied to text columns only. Numeric columns are written as numbers and
 * are never passed through this, so a negative value stays a number rather
 * than becoming the string `'-5`. The JSON export is not touched: it is not a
 * spreadsheet format, and it should carry the user's data byte for byte.
 */
export function neutraliseFormula(value: string): string {
  return /^[=+\-@\t\r]/.test(value) ? `'${value}` : value;
}

function text(value: string | null | undefined): string | null {
  return value == null ? null : neutraliseFormula(value);
}

/**
 * RFC4180-safe field quoting: a field is quoted whenever it contains a
 * comma, a double quote or a line break, and embedded quotes are doubled.
 */
export function csvField(value: unknown): string {
  if (value === null || value === undefined) return '';
  const str = value instanceof Date ? value.toISOString() : String(value);
  if (/[",\r\n]/.test(str)) return `"${str.replace(/"/g, '""')}"`;
  return str;
}

export function csvRow(values: unknown[]): string {
  return values.map(csvField).join(',');
}

const SOURCES: Record<ExportRecordType, RecordSource> = {
  glucose: {
    file: 'glucose',
    jsonKey: 'glucose',
    header: ['id', 'taken_at', 'value_mgdl', 'value_mmoll', 'context', 'note', 'source'],
    async page(userId, from, take) {
      const rows = await prisma.glucoseReading.findMany({
        where: { userId, ...after(from) },
        orderBy: ORDER,
        take,
        // Explicit select: rawPayload (the imported row, possibly large) is
        // never exported and has no business being read a page at a time.
        select: {
          id: true,
          takenAt: true,
          valueMgdl: true,
          context: true,
          note: true,
          source: true,
          deviceId: true,
          importBatchId: true,
          externalId: true,
        },
      });
      return rows.map((r) => {
        const mmoll = round(fromMgdl(r.valueMgdl, 'MMOLL'), 1);
        return {
          position: r,
          json: {
            id: r.id,
            takenAt: r.takenAt.toISOString(),
            valueMgdl: r.valueMgdl,
            valueMmoll: mmoll,
            context: r.context,
            note: r.note,
            source: r.source,
            deviceId: r.deviceId,
            importBatchId: r.importBatchId,
            externalId: r.externalId,
          },
          csv: [
            r.id,
            r.takenAt.toISOString(),
            r.valueMgdl,
            mmoll,
            r.context,
            text(r.note),
            r.source,
          ],
        };
      });
    },
  },
  meal: {
    file: 'meal',
    jsonKey: 'meals',
    header: [
      'id',
      'taken_at',
      'meal_type',
      'description',
      'carbs_g',
      'protein_g',
      'fat_g',
      'fiber_g',
      'calories',
      'portion',
      'note',
      'source',
    ],
    async page(userId, from, take) {
      const rows = await prisma.meal.findMany({
        where: { userId, ...after(from) },
        orderBy: ORDER,
        take,
        include: { foodItems: true },
      });
      return rows.map((r) => ({
        position: r,
        json: {
          id: r.id,
          takenAt: r.takenAt.toISOString(),
          mealType: r.mealType,
          description: r.description,
          carbsG: r.carbsG,
          proteinG: r.proteinG,
          fatG: r.fatG,
          fiberG: r.fiberG,
          calories: r.calories,
          portion: r.portion,
          note: r.note,
          estimateSource: r.estimateSource,
          source: r.source,
          importBatchId: r.importBatchId,
          foodItems: r.foodItems.map((f) => ({
            name: f.name,
            quantity: f.quantity,
            carbsG: f.carbsG,
            proteinG: f.proteinG,
            fatG: f.fatG,
            fiberG: f.fiberG,
            calories: f.calories,
          })),
        },
        csv: [
          r.id,
          r.takenAt.toISOString(),
          r.mealType,
          text(r.description),
          r.carbsG,
          r.proteinG,
          r.fatG,
          r.fiberG,
          r.calories,
          text(r.portion),
          text(r.note),
          r.source,
        ],
      }));
    },
  },
  exercise: {
    file: 'exercise',
    jsonKey: 'exercise',
    header: [
      'id',
      'taken_at',
      'ended_at',
      'activity',
      'duration_min',
      'intensity',
      'distance_km',
      'steps',
      'note',
      'source',
    ],
    async page(userId, from, take) {
      const rows = await prisma.exerciseSession.findMany({
        where: { userId, ...after(from) },
        orderBy: ORDER,
        take,
      });
      return rows.map((r) => ({
        position: r,
        json: {
          id: r.id,
          takenAt: r.takenAt.toISOString(),
          endedAt: r.endedAt?.toISOString() ?? null,
          activity: r.activity,
          durationMin: r.durationMin,
          intensity: r.intensity,
          distanceKm: r.distanceKm,
          steps: r.steps,
          note: r.note,
          source: r.source,
        },
        csv: [
          r.id,
          r.takenAt.toISOString(),
          r.endedAt?.toISOString() ?? '',
          text(r.activity),
          r.durationMin,
          r.intensity,
          r.distanceKm,
          r.steps,
          text(r.note),
          r.source,
        ],
      }));
    },
  },
  sleep: {
    file: 'sleep',
    jsonKey: 'sleep',
    header: ['id', 'bedtime', 'wake_time', 'duration_min', 'quality', 'note', 'source'],
    async page(userId, from, take) {
      const rows = await prisma.sleepSession.findMany({
        where: { userId, ...after(from) },
        orderBy: ORDER,
        take,
      });
      return rows.map((r) => ({
        position: r,
        json: {
          id: r.id,
          takenAt: r.takenAt.toISOString(),
          endedAt: r.endedAt.toISOString(),
          durationMin: r.durationMin,
          quality: r.quality,
          note: r.note,
          source: r.source,
        },
        csv: [
          r.id,
          r.takenAt.toISOString(),
          r.endedAt.toISOString(),
          r.durationMin,
          r.quality,
          text(r.note),
          r.source,
        ],
      }));
    },
  },
  medication: {
    file: 'medication',
    jsonKey: 'medications',
    header: ['id', 'taken_at', 'name', 'dose', 'route', 'note', 'source'],
    async page(userId, from, take) {
      const rows = await prisma.medicationEvent.findMany({
        where: { userId, ...after(from) },
        orderBy: ORDER,
        take,
      });
      return rows.map((r) => ({
        position: r,
        json: {
          id: r.id,
          takenAt: r.takenAt.toISOString(),
          name: r.name,
          dose: r.dose,
          route: r.route,
          note: r.note,
          source: r.source,
        },
        csv: [
          r.id,
          r.takenAt.toISOString(),
          text(r.name),
          text(r.dose),
          text(r.route),
          text(r.note),
          r.source,
        ],
      }));
    },
  },
  weight: {
    file: 'weight',
    jsonKey: 'weight',
    header: ['id', 'taken_at', 'weight_kg', 'note', 'source'],
    async page(userId, from, take) {
      const rows = await prisma.weightMeasurement.findMany({
        where: { userId, ...after(from) },
        orderBy: ORDER,
        take,
      });
      return rows.map((r) => ({
        position: r,
        json: {
          id: r.id,
          takenAt: r.takenAt.toISOString(),
          weightKg: r.weightKg,
          note: r.note,
          source: r.source,
        },
        csv: [r.id, r.takenAt.toISOString(), r.weightKg, text(r.note), r.source],
      }));
    },
  },
  bloodPressure: {
    file: 'blood-pressure',
    jsonKey: 'bloodPressure',
    header: ['id', 'taken_at', 'systolic', 'diastolic', 'pulse', 'note', 'source'],
    async page(userId, from, take) {
      const rows = await prisma.bloodPressureMeasurement.findMany({
        where: { userId, ...after(from) },
        orderBy: ORDER,
        take,
      });
      return rows.map((r) => ({
        position: r,
        json: {
          id: r.id,
          takenAt: r.takenAt.toISOString(),
          systolic: r.systolic,
          diastolic: r.diastolic,
          pulse: r.pulse,
          note: r.note,
          source: r.source,
        },
        csv: [
          r.id,
          r.takenAt.toISOString(),
          r.systolic,
          r.diastolic,
          r.pulse,
          text(r.note),
          r.source,
        ],
      }));
    },
  },
  hydration: {
    file: 'hydration',
    jsonKey: 'hydration',
    header: ['id', 'taken_at', 'volume_ml', 'source'],
    async page(userId, from, take) {
      const rows = await prisma.hydrationEvent.findMany({
        where: { userId, ...after(from) },
        orderBy: ORDER,
        take,
      });
      return rows.map((r) => ({
        position: r,
        json: {
          id: r.id,
          takenAt: r.takenAt.toISOString(),
          volumeMl: r.volumeMl,
          source: r.source,
        },
        csv: [r.id, r.takenAt.toISOString(), r.volumeMl, r.source],
      }));
    },
  },
  symptom: {
    file: 'symptom',
    jsonKey: 'symptoms',
    header: ['id', 'taken_at', 'symptom', 'severity', 'note', 'source'],
    async page(userId, from, take) {
      const rows = await prisma.symptomEntry.findMany({
        where: { userId, ...after(from) },
        orderBy: ORDER,
        take,
      });
      return rows.map((r) => ({
        position: r,
        json: {
          id: r.id,
          takenAt: r.takenAt.toISOString(),
          symptom: r.symptom,
          severity: r.severity,
          note: r.note,
          source: r.source,
        },
        csv: [r.id, r.takenAt.toISOString(), text(r.symptom), r.severity, text(r.note), r.source],
      }));
    },
  },
  mood: {
    file: 'mood',
    jsonKey: 'moods',
    header: ['id', 'taken_at', 'mood', 'stress', 'note', 'source'],
    async page(userId, from, take) {
      const rows = await prisma.moodEntry.findMany({
        where: { userId, ...after(from) },
        orderBy: ORDER,
        take,
      });
      return rows.map((r) => ({
        position: r,
        json: {
          id: r.id,
          takenAt: r.takenAt.toISOString(),
          mood: r.mood,
          stress: r.stress,
          note: r.note,
          source: r.source,
        },
        csv: [r.id, r.takenAt.toISOString(), r.mood, r.stress, text(r.note), r.source],
      }));
    },
  },
  note: {
    file: 'note',
    jsonKey: 'notes',
    header: ['id', 'taken_at', 'text', 'source'],
    async page(userId, from, take) {
      const rows = await prisma.noteEntry.findMany({
        where: { userId, ...after(from) },
        orderBy: ORDER,
        take,
      });
      return rows.map((r) => ({
        position: r,
        json: { id: r.id, takenAt: r.takenAt.toISOString(), text: r.text, source: r.source },
        csv: [r.id, r.takenAt.toISOString(), text(r.text), r.source],
      }));
    },
  },
};

/**
 * Every row of one record type for one user, a page at a time, in (takenAt, id)
 * order. `firstPage`, when given, is the already-read first page (see
 * openJsonExport) and is used instead of querying it again.
 */
async function* pages(
  source: RecordSource,
  userId: string,
  pageSize: number,
  firstPage?: ExportRow[],
): AsyncGenerator<ExportRow[]> {
  let position: Position | null = null;
  let prefetched = firstPage;
  for (;;) {
    const page: ExportRow[] = prefetched ?? (await source.page(userId, position, pageSize));
    prefetched = undefined;
    if (page.length > 0) yield page;
    if (page.length < pageSize) return;
    position = page[page.length - 1]!.position;
  }
}

async function collect(chunks: AsyncIterable<string>): Promise<string> {
  let out = '';
  for await (const chunk of chunks) out += chunk;
  return out;
}

// -------------------------------------------------------------------- JSON

/**
 * Start a JSON export: read the profile and the first page of records now,
 * and return the stream of the rest.
 *
 * Why: a generator runs nothing until its first `next()`, which for a
 * streamed download happens after the route has already committed a 200. A
 * missing profile or a database that is down then surfaced as a 200 with an
 * aborted body instead of an error status. Awaiting this before building the
 * Response moves the reads most likely to fail (the first touch of the
 * database, and the one `OrThrow`) ahead of the status line, where they can
 * still become an error response. A failure on a later page still aborts the
 * stream, which is the best a streamed body can do.
 */
export async function openJsonExport(
  userId: string,
  options: ExportOptions = {},
): Promise<AsyncGenerator<string>> {
  const pageSize = options.pageSize ?? EXPORT_PAGE_SIZE;
  const profile = await prisma.profile.findUniqueOrThrow({ where: { userId } });
  const firstType = EXPORT_RECORD_TYPES[0]!;
  const firstPage = await SOURCES[firstType].page(userId, null, pageSize);
  return jsonChunks(userId, pageSize, profile, firstPage);
}

/**
 * The complete, versioned JSON export for one user, as a stream of text
 * chunks (one per page of records). Pretty-printed at the top level; each
 * record is one compact line, which keeps a multi-year CGM history to a size
 * a person can still download and open. Prefer `openJsonExport` where a
 * failure should surface before the stream starts.
 */
export async function* streamJsonExport(
  userId: string,
  options: ExportOptions = {},
): AsyncGenerator<string> {
  yield* await openJsonExport(userId, options);
}

async function* jsonChunks(
  userId: string,
  pageSize: number,
  profile: Profile,
  firstPage: ExportRow[],
): AsyncGenerator<string> {
  const indent = (value: unknown, depth: number) =>
    JSON.stringify(value, null, 2).replace(/\n/g, `\n${'  '.repeat(depth)}`);

  const head: Omit<JsonExport, 'records' | 'devices' | 'importBatches' | 'meta'> = {
    exportedAt: new Date().toISOString(),
    schemaVersion: EXPORT_SCHEMA_VERSION,
    rowLimit: null,
    profile: {
      displayName: profile.displayName,
      glucoseUnit: profile.glucoseUnit,
      locale: profile.locale,
      timezone: profile.timezone,
      condition: profile.condition,
      targetLowMgdl: profile.targetLowMgdl,
      targetHighMgdl: profile.targetHighMgdl,
      targetLowMmoll: round(fromMgdl(profile.targetLowMgdl, 'MMOLL'), 1),
      targetHighMmoll: round(fromMgdl(profile.targetHighMgdl, 'MMOLL'), 1),
      goals: profile.goals,
      detailLevel: profile.detailLevel,
      createdAt: profile.createdAt.toISOString(),
    },
  };
  yield '{\n';
  for (const [key, value] of Object.entries(head)) {
    yield `  ${JSON.stringify(key)}: ${indent(value, 1)},\n`;
  }

  const meta: Partial<Record<ExportRecordType, ExportMeta>> = {};
  yield '  "records": {\n';
  for (const [index, type] of EXPORT_RECORD_TYPES.entries()) {
    const source = SOURCES[type];
    let count = 0;
    yield `    ${JSON.stringify(source.jsonKey)}: [`;
    const prefetched = index === 0 ? firstPage : undefined;
    for await (const page of pages(source, userId, pageSize, prefetched)) {
      const lines = page.map((row) => `\n      ${JSON.stringify(row.json)}`);
      yield (count > 0 ? ',' : '') + lines.join(',');
      count += page.length;
    }
    meta[type] = { truncated: false, count };
    yield `${count > 0 ? '\n    ' : ''}]${index < EXPORT_RECORD_TYPES.length - 1 ? ',' : ''}\n`;
  }
  yield '  },\n';

  // Devices and import batches are a handful of rows per account (imports are
  // rate limited), so these two are read in one query each.
  const devices = await prisma.device.findMany({
    where: { userId },
    orderBy: { createdAt: 'asc' },
  });
  const importBatches = await prisma.importBatch.findMany({
    where: { userId },
    orderBy: { createdAt: 'asc' },
  });
  const tail = {
    devices: devices.map((d) => ({
      id: d.id,
      label: d.label,
      vendor: d.vendor,
      model: d.model,
      kind: d.kind,
    })),
    importBatches: importBatches.map((b) => ({
      id: b.id,
      connectorName: b.connectorName,
      filename: b.filename,
      status: b.status,
      rowsImported: b.rowsImported,
      rowsDuplicate: b.rowsDuplicate,
      rowsRejected: b.rowsRejected,
      deviceId: b.deviceId,
      createdAt: b.createdAt.toISOString(),
      finishedAt: b.finishedAt?.toISOString() ?? null,
    })),
    meta,
  };
  const entries = Object.entries(tail);
  for (const [index, [key, value]] of entries.entries()) {
    yield `  ${JSON.stringify(key)}: ${indent(value, 1)}${index < entries.length - 1 ? ',' : ''}\n`;
  }
  yield '}\n';
}

/**
 * The whole JSON export as an object. Buffers everything, so the download
 * route streams instead; this exists for tests and small internal callers,
 * and parsing the streamed text (rather than building the object separately)
 * means every test of it is also a test that the stream is valid JSON.
 */
export async function buildJsonExport(
  userId: string,
  options: ExportOptions = {},
): Promise<JsonExport> {
  return JSON.parse(await collect(streamJsonExport(userId, options))) as JsonExport;
}

// -------------------------------------------------------------------- CSV

/** CSV download name (without date or extension) for one record type. */
export function csvExportFilename(type: ExportRecordType): string {
  return `dialog-${SOURCES[type].file}`;
}

async function* csvChunks(
  userId: string,
  type: ExportRecordType,
  pageSize: number,
  firstPage?: ExportRow[],
): AsyncGenerator<{ text: string; rows: number }> {
  const source = SOURCES[type];
  yield { text: csvRow(source.header) + '\r\n', rows: 0 };
  for await (const page of pages(source, userId, pageSize, firstPage)) {
    yield { text: page.map((row) => csvRow(row.csv) + '\r\n').join(''), rows: page.length };
  }
}

/**
 * Start a CSV export: read the first page now and return the stream of the
 * rest, so a database failure throws before a route commits a 200 (see
 * openJsonExport for the full reasoning).
 */
export async function openCsvExport(
  userId: string,
  type: ExportRecordType,
  options: ExportOptions = {},
): Promise<AsyncGenerator<string>> {
  const pageSize = options.pageSize ?? EXPORT_PAGE_SIZE;
  const firstPage = await SOURCES[type].page(userId, null, pageSize);
  return (async function* () {
    for await (const chunk of csvChunks(userId, type, pageSize, firstPage)) yield chunk.text;
  })();
}

/** An RFC4180 CSV for a single record type, scoped to one user, as a stream of text chunks. */
export async function* streamCsvExport(
  userId: string,
  type: ExportRecordType,
  options: ExportOptions = {},
): AsyncGenerator<string> {
  yield* await openCsvExport(userId, type, options);
}

export interface CsvExport {
  filename: string;
  csv: string;
  /** Always false: CSV exports are complete. Kept for existing callers. */
  truncated: boolean;
  rowCount: number;
}

/** The whole CSV in memory. As with buildJsonExport, the route streams instead. */
export async function buildCsvExport(
  userId: string,
  type: ExportRecordType,
  options: ExportOptions = {},
): Promise<CsvExport> {
  let csv = '';
  let rowCount = 0;
  for await (const chunk of csvChunks(userId, type, options.pageSize ?? EXPORT_PAGE_SIZE)) {
    csv += chunk.text;
    rowCount += chunk.rows;
  }
  return { filename: csvExportFilename(type), csv, truncated: false, rowCount };
}

export function exportLabel(type: ExportRecordType): string {
  const labels: Record<ExportRecordType, string> = {
    glucose: 'Glucose readings',
    meal: 'Meals',
    exercise: 'Activity',
    sleep: 'Sleep',
    medication: 'Medications',
    weight: 'Weight',
    bloodPressure: 'Blood pressure',
    hydration: 'Hydration',
    symptom: 'Symptoms',
    mood: 'Mood',
    note: 'Notes',
  };
  return labels[type];
}
