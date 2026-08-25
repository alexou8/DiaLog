/**
 * Omron Connect export connector.
 *
 * Omron's consumer devices (blood pressure monitors and scales, primarily —
 * Omron does not sell a blood glucose meter in most markets) sync to the
 * Omron Connect mobile app, which offers an in-app "export measurement
 * data" action producing a CSV (see docs/DEVICE_INTEGRATIONS.md for sources
 * checked). Omron has not published a formal column-layout spec for this
 * export, and it has been reported to vary by region/app version, so this
 * connector matches tolerantly on the columns consistently reported by
 * users: a measurement date/time column and systolic/diastolic/pulse (blood
 * pressure) or weight columns. Glucose columns are handled too, defensively,
 * in case a regional Omron export or a re-exported/converted file includes
 * them, but this is not verified against an official Omron glucose meter
 * export.
 */
import { parseTimestamp } from '../parse';
import { normalizeGlucoseValue } from '../normalize';
import { inferGlucoseUnit } from '@/lib/domain/units';
import type {
  DeviceConnector,
  GlucoseUnit,
  NormalizedRecord,
  ParseOptions,
  ParseResult,
  ParsedFile,
  RowIssue,
} from '../types';

function norm(h: string): string {
  return h
    .toLowerCase()
    .trim()
    .replace(/[_./]+/g, ' ')
    .replace(/[()]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function findCol(headers: string[], aliases: string[]): number {
  const normalized = headers.map(norm);
  for (const alias of aliases) {
    const idx = normalized.indexOf(alias);
    if (idx >= 0) return idx;
  }
  for (let i = 0; i < normalized.length; i++) {
    const h = normalized[i] ?? '';
    if (aliases.some((a) => h.includes(a))) return i;
  }
  return -1;
}

const ALIASES = {
  date: ['date', 'measurement date'],
  time: ['time', 'measurement time'],
  datetime: ['date time', 'measured at', 'timestamp'],
  systolic: ['sys', 'systolic', 'sys mmhg'],
  diastolic: ['dia', 'diastolic', 'dia mmhg'],
  pulse: ['pulse', 'pulse bpm', 'heart rate'],
  weight: ['weight', 'weight kg', 'body weight'],
  glucose: ['glucose', 'blood glucose', 'bg'],
};

function rawObject(headers: string[], row: string[]): Record<string, unknown> {
  const obj: Record<string, unknown> = {};
  headers.forEach((h, i) => {
    obj[h] = row[i] ?? '';
  });
  return obj;
}

export const omronConnector: DeviceConnector = {
  id: 'omron',
  name: 'Omron Connect',
  vendor: 'Omron',
  description:
    'Reads CSV data exported from the Omron Connect app ("Export measurement data"), covering blood pressure and weight/body-composition measurements. Omron has not published an official export spec; this connector matches the column names consistently reported by users.',
  howToExport: [
    'Open the Omron Connect app and go to History.',
    'Tap the share/export icon, or the "···" menu on a graph screen, and choose "Export measurement data".',
    'Choose CSV format and the date range you want.',
    'Upload the exported CSV file here.',
  ],
  acceptedExtensions: ['.csv'],
  kind: 'BLOOD_PRESSURE_MONITOR',

  detect(sample: ParsedFile): number {
    const rows = sample.rows;
    if (!rows || rows.length < 1) return 0;
    const headers = rows[0] ?? [];
    const hasBpOrWeight =
      (findCol(headers, ALIASES.systolic) >= 0 && findCol(headers, ALIASES.diastolic) >= 0) ||
      findCol(headers, ALIASES.weight) >= 0;
    const hasDate = findCol(headers, ALIASES.date) >= 0 || findCol(headers, ALIASES.datetime) >= 0;
    if (!hasBpOrWeight || !hasDate) return 0;
    return 0.7;
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
    const dataRows = rows.slice(1);

    const iDate = findCol(headers, ALIASES.date);
    const iTime = findCol(headers, ALIASES.time);
    const iDatetime = findCol(headers, ALIASES.datetime);
    const iSys = findCol(headers, ALIASES.systolic);
    const iDia = findCol(headers, ALIASES.diastolic);
    const iPulse = findCol(headers, ALIASES.pulse);
    const iWeight = findCol(headers, ALIASES.weight);
    const iGlucose = findCol(headers, ALIASES.glucose);

    const records: NormalizedRecord[] = [];
    const issues: RowIssue[] = [];
    const warnings: string[] = [];

    let glucoseUnit: GlucoseUnit | null = null;
    if (iGlucose >= 0) {
      const values = dataRows
        .map((r) => Number((r[iGlucose] ?? '').trim()))
        .filter((v) => Number.isFinite(v));
      glucoseUnit = options.unit ?? inferGlucoseUnit(values);
    }

    dataRows.forEach((row, i) => {
      const rowNumber = i + 2;
      const rawRow = row.join(',');
      if (row.every((c) => (c ?? '').trim() === '')) return;

      const tsCell =
        iDatetime >= 0
          ? (row[iDatetime] ?? '')
          : `${iDate >= 0 ? (row[iDate] ?? '') : ''} ${iTime >= 0 ? (row[iTime] ?? '') : ''}`.trim();
      const ts = parseTimestamp(tsCell || null, {
        timezone: options.timezone,
        dateOrder: options.dateOrder,
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

      const raw = rawObject(headers, row);
      let produced = false;

      const sysRaw = iSys >= 0 ? (row[iSys] ?? '').trim() : '';
      const diaRaw = iDia >= 0 ? (row[iDia] ?? '').trim() : '';
      if (sysRaw !== '' && diaRaw !== '') {
        const sys = Number(sysRaw);
        const dia = Number(diaRaw);
        const pulseRaw = iPulse >= 0 ? (row[iPulse] ?? '').trim() : '';
        if (Number.isFinite(sys) && Number.isFinite(dia)) {
          records.push({
            kind: 'note',
            takenAt: ts.date,
            text: `Blood pressure ${sys}/${dia}${pulseRaw ? ` mmHg, pulse ${pulseRaw}` : ' mmHg'}`,
            raw,
          });
          produced = true;
        } else {
          issues.push({
            rowNumber,
            code: 'INVALID_VALUE',
            message: 'Systolic/diastolic value is not a valid number.',
            rawRow,
          });
        }
      }

      const weightRaw = iWeight >= 0 ? (row[iWeight] ?? '').trim() : '';
      if (weightRaw !== '') {
        const w = Number(weightRaw);
        if (Number.isFinite(w) && w > 0) {
          records.push({ kind: 'weight', takenAt: ts.date, weightKg: w, raw });
          produced = true;
        } else {
          issues.push({
            rowNumber,
            code: 'INVALID_VALUE',
            message: 'Weight value is not a valid number.',
            rawRow,
          });
        }
      }

      const glucoseRaw = iGlucose >= 0 ? (row[iGlucose] ?? '').trim() : '';
      if (glucoseRaw !== '') {
        const normalized = normalizeGlucoseValue(Number(glucoseRaw), glucoseUnit);
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
          produced = true;
        }
      }

      if (!produced) {
        issues.push({
          rowNumber,
          code: 'UNSUPPORTED_ROW',
          message: 'Row had no recognised blood pressure, weight, or glucose value.',
          rawRow,
        });
      }
    });

    return {
      records,
      issues,
      rowsTotal: dataRows.filter((r) => !r.every((c) => (c ?? '').trim() === '')).length,
      detectedUnit: glucoseUnit,
      warnings,
    };
  },
};
