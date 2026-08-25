/**
 * Builds a user's full data export.
 *
 * Every query here is scoped by `userId` — there is no code path in this
 * file that can read another account's data. Passwords and other internal
 * secrets are never touched; ids are kept on records (so a user can
 * cross-reference an import batch or a device) but dropped where they carry
 * no meaning outside DiaLog (e.g. the Profile row's own id).
 */
import { prisma } from '@/lib/db/prisma';
import { fromMgdl } from '@/lib/domain/units';

/** Per-record-type cap so a single export request cannot run unbounded. */
export const EXPORT_ROW_LIMIT = 20_000;

export const EXPORT_SCHEMA_VERSION = 1;

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
  /** True when this record type's row count met EXPORT_ROW_LIMIT and was truncated. */
  truncated: boolean;
  count: number;
}

export interface JsonExport {
  exportedAt: string;
  schemaVersion: number;
  rowLimit: number;
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
  meta: Partial<Record<ExportRecordType, ExportMeta>>;
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
}

function withMeta<T>(rows: T[]): { rows: T[]; meta: ExportMeta } {
  const truncated = rows.length > EXPORT_ROW_LIMIT;
  return {
    rows: truncated ? rows.slice(0, EXPORT_ROW_LIMIT) : rows,
    meta: { truncated, count: rows.length },
  };
}

/** Build the complete, versioned JSON export for one user. */
export async function buildJsonExport(userId: string): Promise<JsonExport> {
  const take = EXPORT_ROW_LIMIT + 1; // +1 so we can detect truncation without a second count query.

  const [
    profile,
    glucose,
    meals,
    exercise,
    sleep,
    medications,
    weight,
    bloodPressure,
    hydration,
    symptoms,
    moods,
    notes,
    devices,
    importBatches,
  ] = await Promise.all([
    prisma.profile.findUniqueOrThrow({ where: { userId } }),
    prisma.glucoseReading.findMany({ where: { userId }, orderBy: { takenAt: 'asc' }, take }),
    prisma.meal.findMany({
      where: { userId },
      orderBy: { takenAt: 'asc' },
      take,
      include: { foodItems: true },
    }),
    prisma.exerciseSession.findMany({ where: { userId }, orderBy: { takenAt: 'asc' }, take }),
    prisma.sleepSession.findMany({ where: { userId }, orderBy: { takenAt: 'asc' }, take }),
    prisma.medicationEvent.findMany({ where: { userId }, orderBy: { takenAt: 'asc' }, take }),
    prisma.weightMeasurement.findMany({ where: { userId }, orderBy: { takenAt: 'asc' }, take }),
    prisma.bloodPressureMeasurement.findMany({
      where: { userId },
      orderBy: { takenAt: 'asc' },
      take,
    }),
    prisma.hydrationEvent.findMany({ where: { userId }, orderBy: { takenAt: 'asc' }, take }),
    prisma.symptomEntry.findMany({ where: { userId }, orderBy: { takenAt: 'asc' }, take }),
    prisma.moodEntry.findMany({ where: { userId }, orderBy: { takenAt: 'asc' }, take }),
    prisma.noteEntry.findMany({ where: { userId }, orderBy: { takenAt: 'asc' }, take }),
    prisma.device.findMany({ where: { userId } }),
    prisma.importBatch.findMany({ where: { userId }, orderBy: { createdAt: 'asc' } }),
  ]);

  const glucoseCapped = withMeta(glucose);
  const mealsCapped = withMeta(meals);
  const exerciseCapped = withMeta(exercise);
  const sleepCapped = withMeta(sleep);
  const medicationsCapped = withMeta(medications);
  const weightCapped = withMeta(weight);
  const bpCapped = withMeta(bloodPressure);
  const hydrationCapped = withMeta(hydration);
  const symptomsCapped = withMeta(symptoms);
  const moodsCapped = withMeta(moods);
  const notesCapped = withMeta(notes);

  return {
    exportedAt: new Date().toISOString(),
    schemaVersion: EXPORT_SCHEMA_VERSION,
    rowLimit: EXPORT_ROW_LIMIT,
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
    meta: {
      glucose: glucoseCapped.meta,
      meal: mealsCapped.meta,
      exercise: exerciseCapped.meta,
      sleep: sleepCapped.meta,
      medication: medicationsCapped.meta,
      weight: weightCapped.meta,
      bloodPressure: bpCapped.meta,
      hydration: hydrationCapped.meta,
      symptom: symptomsCapped.meta,
      mood: moodsCapped.meta,
      note: notesCapped.meta,
    },
    records: {
      glucose: glucoseCapped.rows.map((r) => ({
        id: r.id,
        takenAt: r.takenAt.toISOString(),
        valueMgdl: r.valueMgdl,
        valueMmoll: round(fromMgdl(r.valueMgdl, 'MMOLL'), 1),
        context: r.context,
        note: r.note,
        source: r.source,
        deviceId: r.deviceId,
        importBatchId: r.importBatchId,
        externalId: r.externalId,
      })),
      meals: mealsCapped.rows.map((r) => ({
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
      })),
      exercise: exerciseCapped.rows.map((r) => ({
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
      })),
      sleep: sleepCapped.rows.map((r) => ({
        id: r.id,
        takenAt: r.takenAt.toISOString(),
        endedAt: r.endedAt.toISOString(),
        durationMin: r.durationMin,
        quality: r.quality,
        note: r.note,
        source: r.source,
      })),
      medications: medicationsCapped.rows.map((r) => ({
        id: r.id,
        takenAt: r.takenAt.toISOString(),
        name: r.name,
        dose: r.dose,
        route: r.route,
        note: r.note,
        source: r.source,
      })),
      weight: weightCapped.rows.map((r) => ({
        id: r.id,
        takenAt: r.takenAt.toISOString(),
        weightKg: r.weightKg,
        note: r.note,
        source: r.source,
      })),
      bloodPressure: bpCapped.rows.map((r) => ({
        id: r.id,
        takenAt: r.takenAt.toISOString(),
        systolic: r.systolic,
        diastolic: r.diastolic,
        pulse: r.pulse,
        note: r.note,
        source: r.source,
      })),
      hydration: hydrationCapped.rows.map((r) => ({
        id: r.id,
        takenAt: r.takenAt.toISOString(),
        volumeMl: r.volumeMl,
        source: r.source,
      })),
      symptoms: symptomsCapped.rows.map((r) => ({
        id: r.id,
        takenAt: r.takenAt.toISOString(),
        symptom: r.symptom,
        severity: r.severity,
        note: r.note,
        source: r.source,
      })),
      moods: moodsCapped.rows.map((r) => ({
        id: r.id,
        takenAt: r.takenAt.toISOString(),
        mood: r.mood,
        stress: r.stress,
        note: r.note,
        source: r.source,
      })),
      notes: notesCapped.rows.map((r) => ({
        id: r.id,
        takenAt: r.takenAt.toISOString(),
        text: r.text,
        source: r.source,
      })),
    },
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
  };
}

function round(value: number, digits: number): number {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

// -------------------------------------------------------------------- CSV

/**
 * RFC4180-safe field quoting: a field is quoted whenever it contains a
 * comma, a double quote or a line break, and embedded quotes are doubled.
 */
function csvField(value: unknown): string {
  if (value === null || value === undefined) return '';
  const str = value instanceof Date ? value.toISOString() : String(value);
  if (/[",\r\n]/.test(str)) return `"${str.replace(/"/g, '""')}"`;
  return str;
}

function csvRow(values: unknown[]): string {
  return values.map(csvField).join(',');
}

export interface CsvExport {
  filename: string;
  csv: string;
  truncated: boolean;
  rowCount: number;
}

/** Build an RFC4180 CSV for a single record type, scoped to one user. */
export async function buildCsvExport(userId: string, type: ExportRecordType): Promise<CsvExport> {
  const take = EXPORT_ROW_LIMIT + 1;

  switch (type) {
    case 'glucose': {
      const rows = await prisma.glucoseReading.findMany({
        where: { userId },
        orderBy: { takenAt: 'asc' },
        take,
      });
      const { rows: capped, meta } = withMeta(rows);
      const header = ['id', 'taken_at', 'value_mgdl', 'value_mmoll', 'context', 'note', 'source'];
      const lines = capped.map((r) =>
        csvRow([
          r.id,
          r.takenAt.toISOString(),
          r.valueMgdl,
          round(fromMgdl(r.valueMgdl, 'MMOLL'), 1),
          r.context,
          r.note,
          r.source,
        ]),
      );
      return finish('glucose', header, lines, meta);
    }
    case 'meal': {
      const rows = await prisma.meal.findMany({
        where: { userId },
        orderBy: { takenAt: 'asc' },
        take,
      });
      const { rows: capped, meta } = withMeta(rows);
      const header = [
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
      ];
      const lines = capped.map((r) =>
        csvRow([
          r.id,
          r.takenAt.toISOString(),
          r.mealType,
          r.description,
          r.carbsG,
          r.proteinG,
          r.fatG,
          r.fiberG,
          r.calories,
          r.portion,
          r.note,
          r.source,
        ]),
      );
      return finish('meal', header, lines, meta);
    }
    case 'exercise': {
      const rows = await prisma.exerciseSession.findMany({
        where: { userId },
        orderBy: { takenAt: 'asc' },
        take,
      });
      const { rows: capped, meta } = withMeta(rows);
      const header = [
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
      ];
      const lines = capped.map((r) =>
        csvRow([
          r.id,
          r.takenAt.toISOString(),
          r.endedAt?.toISOString() ?? '',
          r.activity,
          r.durationMin,
          r.intensity,
          r.distanceKm,
          r.steps,
          r.note,
          r.source,
        ]),
      );
      return finish('exercise', header, lines, meta);
    }
    case 'sleep': {
      const rows = await prisma.sleepSession.findMany({
        where: { userId },
        orderBy: { takenAt: 'asc' },
        take,
      });
      const { rows: capped, meta } = withMeta(rows);
      const header = ['id', 'bedtime', 'wake_time', 'duration_min', 'quality', 'note', 'source'];
      const lines = capped.map((r) =>
        csvRow([
          r.id,
          r.takenAt.toISOString(),
          r.endedAt.toISOString(),
          r.durationMin,
          r.quality,
          r.note,
          r.source,
        ]),
      );
      return finish('sleep', header, lines, meta);
    }
    case 'medication': {
      const rows = await prisma.medicationEvent.findMany({
        where: { userId },
        orderBy: { takenAt: 'asc' },
        take,
      });
      const { rows: capped, meta } = withMeta(rows);
      const header = ['id', 'taken_at', 'name', 'dose', 'route', 'note', 'source'];
      const lines = capped.map((r) =>
        csvRow([r.id, r.takenAt.toISOString(), r.name, r.dose, r.route, r.note, r.source]),
      );
      return finish('medication', header, lines, meta);
    }
    case 'weight': {
      const rows = await prisma.weightMeasurement.findMany({
        where: { userId },
        orderBy: { takenAt: 'asc' },
        take,
      });
      const { rows: capped, meta } = withMeta(rows);
      const header = ['id', 'taken_at', 'weight_kg', 'note', 'source'];
      const lines = capped.map((r) =>
        csvRow([r.id, r.takenAt.toISOString(), r.weightKg, r.note, r.source]),
      );
      return finish('weight', header, lines, meta);
    }
    case 'bloodPressure': {
      const rows = await prisma.bloodPressureMeasurement.findMany({
        where: { userId },
        orderBy: { takenAt: 'asc' },
        take,
      });
      const { rows: capped, meta } = withMeta(rows);
      const header = ['id', 'taken_at', 'systolic', 'diastolic', 'pulse', 'note', 'source'];
      const lines = capped.map((r) =>
        csvRow([r.id, r.takenAt.toISOString(), r.systolic, r.diastolic, r.pulse, r.note, r.source]),
      );
      return finish('blood-pressure', header, lines, meta);
    }
    case 'hydration': {
      const rows = await prisma.hydrationEvent.findMany({
        where: { userId },
        orderBy: { takenAt: 'asc' },
        take,
      });
      const { rows: capped, meta } = withMeta(rows);
      const header = ['id', 'taken_at', 'volume_ml', 'source'];
      const lines = capped.map((r) =>
        csvRow([r.id, r.takenAt.toISOString(), r.volumeMl, r.source]),
      );
      return finish('hydration', header, lines, meta);
    }
    case 'symptom': {
      const rows = await prisma.symptomEntry.findMany({
        where: { userId },
        orderBy: { takenAt: 'asc' },
        take,
      });
      const { rows: capped, meta } = withMeta(rows);
      const header = ['id', 'taken_at', 'symptom', 'severity', 'note', 'source'];
      const lines = capped.map((r) =>
        csvRow([r.id, r.takenAt.toISOString(), r.symptom, r.severity, r.note, r.source]),
      );
      return finish('symptom', header, lines, meta);
    }
    case 'mood': {
      const rows = await prisma.moodEntry.findMany({
        where: { userId },
        orderBy: { takenAt: 'asc' },
        take,
      });
      const { rows: capped, meta } = withMeta(rows);
      const header = ['id', 'taken_at', 'mood', 'stress', 'note', 'source'];
      const lines = capped.map((r) =>
        csvRow([r.id, r.takenAt.toISOString(), r.mood, r.stress, r.note, r.source]),
      );
      return finish('mood', header, lines, meta);
    }
    case 'note': {
      const rows = await prisma.noteEntry.findMany({
        where: { userId },
        orderBy: { takenAt: 'asc' },
        take,
      });
      const { rows: capped, meta } = withMeta(rows);
      const header = ['id', 'taken_at', 'text', 'source'];
      const lines = capped.map((r) => csvRow([r.id, r.takenAt.toISOString(), r.text, r.source]));
      return finish('note', header, lines, meta);
    }
  }
}

function finish(name: string, header: string[], lines: string[], meta: ExportMeta): CsvExport {
  const csv = [csvRow(header), ...lines].join('\r\n') + '\r\n';
  return { filename: `dialog-${name}`, csv, truncated: meta.truncated, rowCount: meta.count };
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
