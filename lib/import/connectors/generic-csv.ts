/**
 * Generic, column-name-driven CSV connector.
 *
 * This is the fallback connector: it understands no particular vendor's
 * layout, only common column names (case/space/underscore-insensitive, with
 * a handful of well-known aliases). It is what makes "export a CSV from
 * whatever app you use and upload it" work even for devices we have no
 * dedicated connector for.
 */
import { mapGlucoseContext, normalizeGlucoseValue } from '../normalize';
import { parseTimestamp, resolveDateOrder } from '../parse';
import type {
  DeviceConnector,
  GlucoseUnit,
  NormalizedRecord,
  ParseOptions,
  ParseResult,
  ParsedFile,
  RowIssue,
} from '../types';
import { inferGlucoseUnit } from '@/lib/domain/units';

function normalizeHeader(h: string): string {
  return h
    .toLowerCase()
    .trim()
    .replace(/[_./]+/g, ' ')
    .replace(/[()]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

const ALIASES = {
  timestamp: ['timestamp', 'datetime', 'date time', 'time stamp', 'device timestamp'],
  date: ['date'],
  time: ['time'],
  glucose: [
    'glucose',
    'blood glucose',
    'bg',
    'glucose value',
    'glucose reading',
    'reading',
    'value',
    'glucose level',
    'glucose mg dl',
    'glucose mmol l',
    'blood glucose mg dl',
    'blood glucose mmol l',
  ],
  unit: ['unit', 'units', 'glucose unit'],
  context: ['context', 'tag', 'meal context', 'meal tag'],
  notes: ['notes', 'note', 'comment', 'comments'],
  carbs: [
    'carbs',
    'carbohydrates',
    'carbs g',
    'carbohydrates g',
    'carb grams',
    'carbohydrates grams',
  ],
};

function findColumn(headers: string[], aliases: string[]): number {
  const normalized = headers.map(normalizeHeader);
  for (const alias of aliases) {
    const idx = normalized.indexOf(alias);
    if (idx >= 0) return idx;
  }
  for (let i = 0; i < normalized.length; i++) {
    const h = normalized[i] ?? '';
    if (aliases.some((alias) => h.includes(alias))) return i;
  }
  return -1;
}

function unitFromHeaderText(header: string): GlucoseUnit | null {
  const h = header.toLowerCase();
  if (h.includes('mmol')) return 'MMOLL';
  if (h.includes('mg/dl') || h.includes('mg dl') || h.includes('mgdl')) return 'MGDL';
  return null;
}

interface Columns {
  timestamp: number;
  date: number;
  time: number;
  glucose: number;
  unit: number;
  context: number;
  notes: number;
  carbs: number;
}

function locateColumns(headers: string[]): Columns {
  return {
    timestamp: findColumn(headers, ALIASES.timestamp),
    date: findColumn(headers, ALIASES.date),
    time: findColumn(headers, ALIASES.time),
    glucose: findColumn(headers, ALIASES.glucose),
    unit: findColumn(headers, ALIASES.unit),
    context: findColumn(headers, ALIASES.context),
    notes: findColumn(headers, ALIASES.notes),
    carbs: findColumn(headers, ALIASES.carbs),
  };
}

function cell(row: string[], idx: number): string {
  if (idx < 0) return '';
  return (row[idx] ?? '').trim();
}

function rawObject(headers: string[], row: string[]): Record<string, unknown> {
  const obj: Record<string, unknown> = {};
  headers.forEach((h, i) => {
    obj[h] = row[i] ?? '';
  });
  return obj;
}

export const genericCsvConnector: DeviceConnector = {
  id: 'generic-csv',
  name: 'Generic CSV',
  vendor: 'Generic',
  description:
    'Column-name-driven CSV import for exports from apps and meters this importer has no dedicated connector for. Recognises common column names such as "glucose", "bg", "date"/"time" or "timestamp", "carbs", and "notes".',
  howToExport: [
    'Export your data as a CSV file from your app or device software.',
    'Make sure the file has a header row naming each column.',
    'Upload the CSV file here.',
  ],
  acceptedExtensions: ['.csv', '.tsv', '.txt'],
  kind: 'OTHER',

  detect(sample: ParsedFile): number {
    const rows = sample.rows;
    if (!rows || rows.length < 1) return 0;
    const headers = rows[0] ?? [];
    const cols = locateColumns(headers);
    if (cols.glucose < 0) return 0;
    if (cols.timestamp < 0 && cols.date < 0) return 0;
    return 0.5;
  },

  parse(file: ParsedFile, options: ParseOptions): ParseResult {
    return parseGenericCsv(file, options);
  },
};

/** Exported so dialog-legacy and other thin wrappers can reuse the row-walking logic if useful. */
export function parseGenericCsv(file: ParsedFile, options: ParseOptions): ParseResult {
  const rows = file.rows ?? [];
  const issues: RowIssue[] = [];
  const warnings: string[] = [];
  const records: NormalizedRecord[] = [];

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
  const dataRows = rows.slice(1);
  const cols = locateColumns(headers);

  if (cols.glucose < 0) {
    return {
      records: [],
      issues: dataRows.map((_, i) => ({
        rowNumber: i + 2,
        code: 'UNSUPPORTED_ROW' as const,
        message: 'No recognisable glucose value column found in this file.',
      })),
      rowsTotal: dataRows.length,
      detectedUnit: null,
      warnings: ['Could not find a glucose value column.'],
    };
  }

  const explicitUnit =
    options.unit ?? unitFromHeaderText(cols.glucose >= 0 ? (headers[cols.glucose] ?? '') : '');

  // Pass 1: resolve date order and glucose unit across the whole file.
  const dateStrings: string[] = [];
  const glucoseValues: number[] = [];
  for (const row of dataRows) {
    const tsRaw =
      cols.timestamp >= 0
        ? cell(row, cols.timestamp)
        : `${cell(row, cols.date)} ${cell(row, cols.time)}`.trim();
    if (tsRaw) dateStrings.push(tsRaw);
    const gRaw = cell(row, cols.glucose);
    const g = Number(gRaw);
    if (gRaw && Number.isFinite(g)) glucoseValues.push(g);
  }
  const { order, assumed } = resolveDateOrder(dateStrings, options.dateOrder);
  if (assumed && dateStrings.some((d) => /^\d{1,2}\/\d{1,2}\/\d{4}/.test(d))) {
    warnings.push(
      `Ambiguous day/month order in dates; assumed ${order === 'DMY' ? 'day/month/year' : 'month/day/year'}.`,
    );
  }

  const detectedUnit: GlucoseUnit | null = explicitUnit ?? inferGlucoseUnit(glucoseValues);
  if (!explicitUnit && detectedUnit) {
    warnings.push(
      `Glucose unit not stated in file; inferred ${detectedUnit === 'MMOLL' ? 'mmol/L' : 'mg/dL'} from value magnitudes.`,
    );
  }

  // Pass 2: build records.
  dataRows.forEach((row, i) => {
    const rowNumber = i + 2;
    const rawRow = row.join(',');
    const tsRaw =
      cols.timestamp >= 0
        ? cell(row, cols.timestamp)
        : `${cell(row, cols.date)} ${cell(row, cols.time)}`.trim();
    const ts = parseTimestamp(tsRaw || null, {
      timezone: options.timezone,
      dateOrder: order,
      now: options.now,
    });
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

    const gRaw = cell(row, cols.glucose);
    if (gRaw !== '') {
      const gVal = Number(gRaw);
      const perRowUnit =
        (cols.unit >= 0
          ? (unitFromHeaderText(cell(row, cols.unit)) ??
            (['mmol/l', 'mmoll'].includes(cell(row, cols.unit).toLowerCase())
              ? 'MMOLL'
              : cell(row, cols.unit).toLowerCase().includes('mg')
                ? 'MGDL'
                : null))
          : null) ?? detectedUnit;
      const normalized = normalizeGlucoseValue(gVal, perRowUnit);
      if (normalized.valueMgdl === null) {
        issues.push({
          rowNumber,
          code: normalized.issue ?? 'INVALID_VALUE',
          message: normalized.message ?? 'Invalid glucose value.',
          rawRow,
        });
      } else {
        const notesText = cols.notes >= 0 ? cell(row, cols.notes) : '';
        const contextText = cols.context >= 0 ? cell(row, cols.context) : notesText;
        records.push({
          kind: 'glucose',
          takenAt: ts.date,
          valueMgdl: normalized.valueMgdl,
          context: mapGlucoseContext(contextText),
          note: notesText || undefined,
          raw: rawObject(headers, row),
        });
      }
    } else {
      issues.push({
        rowNumber,
        code: 'MISSING_VALUE',
        message: 'Glucose value column is empty.',
        rawRow,
      });
    }

    if (cols.carbs >= 0) {
      const carbsRaw = cell(row, cols.carbs);
      if (carbsRaw !== '') {
        const carbsVal = Number(carbsRaw);
        if (Number.isFinite(carbsVal)) {
          records.push({
            kind: 'meal',
            takenAt: ts.date,
            description: 'Imported meal',
            mealType: 'OTHER',
            carbsG: carbsVal,
            raw: rawObject(headers, row),
          });
        }
      }
    }
  });

  return { records, issues, rowsTotal: dataRows.length, detectedUnit, warnings };
}
