/**
 * DiaLog's own legacy sample-data formats, as seen in ml/data/*.csv:
 *
 *  - Long/event format (sample_logs.csv): one row per event, discriminated by
 *    `event_type` in {"med","meal","glucose"} —
 *    `timestamp,event_type,carbs_g,med_name,med_units,glucose_mgdl,notes`
 *
 *  - Wide/hourly format (sample_glucose_data.csv): one row per hour with
 *    several co-occurring measurements —
 *    `timestamp,glucose_mg_dl,carbs_grams,insulin_units,activity_minutes,stress_level,sleep_hours`
 *
 * Both predate the general-purpose importer and must keep working exactly as
 * they did when the ML pipeline produced them.
 */
import { parseTimestamp } from '../parse';
import { normalizeGlucoseValue } from '../normalize';
import type {
  DeviceConnector,
  NormalizedRecord,
  ParseOptions,
  ParseResult,
  ParsedFile,
  RowIssue,
} from '../types';

function norm(h: string): string {
  return h.toLowerCase().trim().replace(/\s+/g, '_');
}

const LONG_HEADERS = [
  'timestamp',
  'event_type',
  'carbs_g',
  'med_name',
  'med_units',
  'glucose_mgdl',
  'notes',
];
const WIDE_HEADERS = [
  'timestamp',
  'glucose_mg_dl',
  'carbs_grams',
  'insulin_units',
  'activity_minutes',
  'stress_level',
  'sleep_hours',
];

function headerOverlap(headers: string[], expected: string[]): number {
  const normalized = new Set(headers.map(norm));
  const hits = expected.filter((e) => normalized.has(e)).length;
  return hits / expected.length;
}

function rawObject(headers: string[], row: string[]): Record<string, unknown> {
  const obj: Record<string, unknown> = {};
  headers.forEach((h, i) => {
    obj[h] = row[i] ?? '';
  });
  return obj;
}

export const dialogLegacyConnector: DeviceConnector = {
  id: 'dialog-legacy',
  name: 'DiaLog legacy sample log',
  vendor: 'DiaLog',
  description:
    "Reads DiaLog's own historical sample-data CSV formats: a long per-event log (timestamp, event_type, carbs_g, med_name, med_units, glucose_mgdl, notes) and a wide per-hour table (timestamp, glucose_mg_dl, carbs_grams, insulin_units, activity_minutes, stress_level, sleep_hours).",
  howToExport: [
    "This format is produced internally by DiaLog's ML sample data; no export steps needed.",
  ],
  acceptedExtensions: ['.csv'],
  kind: 'OTHER',

  detect(sample: ParsedFile): number {
    const rows = sample.rows;
    if (!rows || rows.length < 1) return 0;
    const headers = rows[0] ?? [];
    const longScore = headerOverlap(headers, LONG_HEADERS);
    const wideScore = headerOverlap(headers, WIDE_HEADERS);
    const best = Math.max(longScore, wideScore);
    return best >= 0.85 ? 0.97 : 0;
  },

  parse(file: ParsedFile, options: ParseOptions): ParseResult {
    const rows = file.rows ?? [];
    if (rows.length === 0) {
      return {
        records: [],
        issues: [],
        rowsTotal: 0,
        detectedUnit: null,
        warnings: ['File is empty.'],
      };
    }
    const headers = rows[0] ?? [];
    const isLong = headerOverlap(headers, LONG_HEADERS) >= headerOverlap(headers, WIDE_HEADERS);
    return isLong
      ? parseLongFormat(headers, rows.slice(1), options)
      : parseWideFormat(headers, rows.slice(1), options);
  },
};

function colIndex(headers: string[], name: string): number {
  return headers.findIndex((h) => norm(h) === name);
}

function parseLongFormat(
  headers: string[],
  dataRows: string[][],
  options: ParseOptions,
): ParseResult {
  const iTimestamp = colIndex(headers, 'timestamp');
  const iEventType = colIndex(headers, 'event_type');
  const iCarbs = colIndex(headers, 'carbs_g');
  const iMedName = colIndex(headers, 'med_name');
  const iMedUnits = colIndex(headers, 'med_units');
  const iGlucose = colIndex(headers, 'glucose_mgdl');
  const iNotes = colIndex(headers, 'notes');

  const records: NormalizedRecord[] = [];
  const issues: RowIssue[] = [];

  dataRows.forEach((row, i) => {
    const rowNumber = i + 2;
    const rawRow = row.join(',');
    const tsCell = iTimestamp >= 0 ? (row[iTimestamp] ?? '') : '';
    const ts = parseTimestamp(tsCell || null, { timezone: options.timezone, now: options.now });
    if (!ts.date) {
      issues.push({
        rowNumber,
        code: ts.error === 'MISSING_TIMESTAMP' ? 'MISSING_TIMESTAMP' : 'INVALID_TIMESTAMP',
        message: ts.message ?? 'Invalid timestamp.',
        rawRow,
      });
      return;
    }
    if (ts.error === 'FUTURE_TIMESTAMP') {
      issues.push({
        rowNumber,
        code: 'FUTURE_TIMESTAMP',
        message: ts.message ?? 'Timestamp is in the future.',
        rawRow,
      });
      return;
    }

    const eventType = (iEventType >= 0 ? (row[iEventType] ?? '') : '').trim().toLowerCase();
    const raw = rawObject(headers, row);
    const notes = iNotes >= 0 ? (row[iNotes] ?? '').trim() : '';

    if (eventType === 'glucose') {
      const gRaw = iGlucose >= 0 ? (row[iGlucose] ?? '').trim() : '';
      if (gRaw === '') {
        issues.push({
          rowNumber,
          code: 'MISSING_VALUE',
          message: 'glucose event has no glucose_mgdl value.',
          rawRow,
        });
        return;
      }
      const normalized = normalizeGlucoseValue(Number(gRaw), 'MGDL');
      if (normalized.valueMgdl === null) {
        issues.push({
          rowNumber,
          code: normalized.issue ?? 'INVALID_VALUE',
          message: normalized.message ?? 'Invalid glucose value.',
          rawRow,
        });
        return;
      }
      records.push({
        kind: 'glucose',
        takenAt: ts.date,
        valueMgdl: normalized.valueMgdl,
        context: 'UNKNOWN',
        note: notes || undefined,
        raw,
      });
      return;
    }

    if (eventType === 'meal') {
      const carbsRaw = iCarbs >= 0 ? (row[iCarbs] ?? '').trim() : '';
      const carbsG = carbsRaw === '' ? null : Number(carbsRaw);
      records.push({
        kind: 'meal',
        takenAt: ts.date,
        description: notes || 'Imported meal',
        mealType: 'OTHER',
        carbsG: carbsG !== null && Number.isFinite(carbsG) ? carbsG : null,
        raw,
      });
      return;
    }

    if (eventType === 'med') {
      const name = iMedName >= 0 ? (row[iMedName] ?? '').trim() : '';
      const dose = iMedUnits >= 0 ? (row[iMedUnits] ?? '').trim() : '';
      if (!name) {
        issues.push({
          rowNumber,
          code: 'MISSING_VALUE',
          message: 'med event has no med_name value.',
          rawRow,
        });
        return;
      }
      records.push({ kind: 'medication', takenAt: ts.date, name, dose: dose || null, raw });
      return;
    }

    issues.push({
      rowNumber,
      code: 'UNSUPPORTED_ROW',
      message: `Unrecognised event_type "${eventType}".`,
      rawRow,
    });
  });

  return { records, issues, rowsTotal: dataRows.length, detectedUnit: 'MGDL', warnings: [] };
}

function parseWideFormat(
  headers: string[],
  dataRows: string[][],
  options: ParseOptions,
): ParseResult {
  const iTimestamp = colIndex(headers, 'timestamp');
  const iGlucose = colIndex(headers, 'glucose_mg_dl');
  const iCarbs = colIndex(headers, 'carbs_grams');
  const iInsulin = colIndex(headers, 'insulin_units');
  const iActivity = colIndex(headers, 'activity_minutes');

  const records: NormalizedRecord[] = [];
  const issues: RowIssue[] = [];

  dataRows.forEach((row, i) => {
    const rowNumber = i + 2;
    const rawRow = row.join(',');
    const tsCell = iTimestamp >= 0 ? (row[iTimestamp] ?? '') : '';
    const ts = parseTimestamp(tsCell || null, { timezone: options.timezone, now: options.now });
    if (!ts.date) {
      issues.push({
        rowNumber,
        code: ts.error === 'MISSING_TIMESTAMP' ? 'MISSING_TIMESTAMP' : 'INVALID_TIMESTAMP',
        message: ts.message ?? 'Invalid timestamp.',
        rawRow,
      });
      return;
    }
    if (ts.error === 'FUTURE_TIMESTAMP') {
      issues.push({
        rowNumber,
        code: 'FUTURE_TIMESTAMP',
        message: ts.message ?? 'Timestamp is in the future.',
        rawRow,
      });
      return;
    }
    const raw = rawObject(headers, row);

    const gRaw = iGlucose >= 0 ? (row[iGlucose] ?? '').trim() : '';
    if (gRaw === '') {
      issues.push({
        rowNumber,
        code: 'MISSING_VALUE',
        message: 'Row has no glucose_mg_dl value.',
        rawRow,
      });
    } else {
      const normalized = normalizeGlucoseValue(Number(gRaw), 'MGDL');
      if (normalized.valueMgdl === null) {
        issues.push({
          rowNumber,
          code: normalized.issue ?? 'INVALID_VALUE',
          message: normalized.message ?? 'Invalid glucose value.',
          rawRow,
        });
      } else {
        records.push({
          kind: 'glucose',
          takenAt: ts.date,
          valueMgdl: normalized.valueMgdl,
          context: 'UNKNOWN',
          raw,
        });
      }
    }

    const carbsRaw = iCarbs >= 0 ? Number((row[iCarbs] ?? '').trim()) : NaN;
    if (Number.isFinite(carbsRaw) && carbsRaw > 0) {
      records.push({
        kind: 'meal',
        takenAt: ts.date,
        description: 'Imported meal',
        mealType: 'OTHER',
        carbsG: carbsRaw,
        raw,
      });
    }

    const insulinRaw = iInsulin >= 0 ? Number((row[iInsulin] ?? '').trim()) : NaN;
    if (Number.isFinite(insulinRaw) && insulinRaw > 0) {
      records.push({
        kind: 'medication',
        takenAt: ts.date,
        name: 'Insulin',
        dose: `${insulinRaw}`,
        raw,
      });
    }

    const activityRaw = iActivity >= 0 ? Number((row[iActivity] ?? '').trim()) : NaN;
    if (Number.isFinite(activityRaw) && activityRaw > 0) {
      records.push({
        kind: 'exercise',
        takenAt: ts.date,
        activity: 'Activity',
        durationMin: Math.round(activityRaw),
        raw,
      });
    }
  });

  return { records, issues, rowsTotal: dataRows.length, detectedUnit: 'MGDL', warnings: [] };
}
