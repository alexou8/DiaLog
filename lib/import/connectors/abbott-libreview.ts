/**
 * Abbott LibreView / FreeStyle Libre CSV export connector.
 *
 * Layout modelled on the LibreView "Glucose data" CSV export, as documented
 * by community reverse-engineering efforts (Abbott has not published a
 * formal file-format spec for this export — see docs/DEVICE_INTEGRATIONS.md
 * for sources and the date checked). The export begins with a short
 * preamble (device/serial info) before the real header row, and uses a
 * numeric `Record Type` column to discriminate row kinds:
 *
 *   0 = Historic Glucose (sensor's periodic reading)
 *   1 = Scan Glucose (user-initiated scan)
 *   2 = Insulin (non-numeric / rapid-acting / long-acting)
 *   3 = Food / carbohydrates
 *   4 = Notes
 *   5 = Strip Glucose (fingerstick, via the meter built into the reader)
 *   6 = Ketone
 *
 * Column names appear in either mg/dL or mmol/L, e.g. "Historic Glucose
 * mg/dL" or "Historic Glucose mmol/L" — which one is present tells us the
 * file's unit directly, no inference needed. Because the exact set and
 * spelling of columns has changed across LibreView software versions and we
 * could not verify every historical variant, unrecognised columns are
 * ignored defensively rather than causing a parse failure.
 */
import { mapGlucoseContext, normalizeGlucoseValue } from '../normalize';
import { parseTimestamp } from '../parse';
import type {
  DeviceConnector,
  GlucoseContext,
  GlucoseUnit,
  NormalizedRecord,
  ParseOptions,
  ParseResult,
  ParsedFile,
  RowIssue,
} from '../types';

function norm(h: string): string {
  return h.toLowerCase().trim().replace(/\s+/g, ' ');
}

/** Finds the real header row: LibreView exports a short preamble before it. */
function findHeaderRowIndex(rows: string[][]): number {
  for (let i = 0; i < Math.min(rows.length, 5); i++) {
    const row = rows[i] ?? [];
    const normalized = row.map(norm);
    if (
      normalized.some((c) => c === 'device timestamp') &&
      normalized.some((c) => c.includes('record type'))
    ) {
      return i;
    }
  }
  return -1;
}

function findCol(headers: string[], predicate: (h: string) => boolean): number {
  return headers.findIndex((h) => predicate(norm(h)));
}

function rawObject(headers: string[], row: string[]): Record<string, unknown> {
  const obj: Record<string, unknown> = {};
  headers.forEach((h, i) => {
    obj[h] = row[i] ?? '';
  });
  return obj;
}

export const abbottLibreViewConnector: DeviceConnector = {
  id: 'abbott-libreview',
  name: 'Abbott LibreView (FreeStyle Libre)',
  vendor: 'Abbott',
  description:
    'Reads the "Glucose data" CSV export from LibreView (libreview.com), which aggregates FreeStyle Libre sensor scans/history alongside logged insulin, carbs, and notes. Column layout verified against community documentation, not an official Abbott spec.',
  howToExport: [
    'Sign in to LibreView (libreview.com) with the account linked to your FreeStyle Libre sensor(s).',
    'Open the "Reports" or "Download data" section.',
    'Choose "Glucose data" export and download the CSV file.',
    'Upload that CSV file here.',
  ],
  acceptedExtensions: ['.csv'],
  kind: 'CGM',

  detect(sample: ParsedFile): number {
    const rows = sample.rows;
    if (!rows) return 0;
    const idx = findHeaderRowIndex(rows);
    return idx >= 0 ? 0.9 : 0;
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
    const headerIdx = findHeaderRowIndex(rows);
    if (headerIdx < 0) {
      return {
        records: [],
        issues: [],
        rowsTotal: 0,
        detectedUnit: null,
        warnings: ['Could not find the LibreView header row ("Device Timestamp" + "Record Type").'],
      };
    }
    const headers = rows[headerIdx] ?? [];
    const dataRows = rows.slice(headerIdx + 1);

    const iTimestamp = findCol(headers, (h) => h === 'device timestamp');
    const iRecordType = findCol(headers, (h) => h.includes('record type'));
    const iHistoricMgdl = findCol(
      headers,
      (h) => h.includes('historic glucose') && h.includes('mg/dl'),
    );
    const iHistoricMmol = findCol(
      headers,
      (h) => h.includes('historic glucose') && h.includes('mmol'),
    );
    const iScanMgdl = findCol(headers, (h) => h.includes('scan glucose') && h.includes('mg/dl'));
    const iScanMmol = findCol(headers, (h) => h.includes('scan glucose') && h.includes('mmol'));
    const iStripMgdl = findCol(headers, (h) => h.includes('strip glucose') && h.includes('mg/dl'));
    const iStripMmol = findCol(headers, (h) => h.includes('strip glucose') && h.includes('mmol'));
    const iCarbs = findCol(headers, (h) => h.includes('carbohydrate'));
    const iNotes = findCol(headers, (h) => h === 'notes');
    const iRapidInsulin = findCol(
      headers,
      (h) => h.includes('rapid-acting insulin') && !h.includes('non-numeric'),
    );
    const iLongInsulin = findCol(
      headers,
      (h) => h.includes('long-acting insulin') && !h.includes('non-numeric'),
    );

    const unit: GlucoseUnit | null =
      options.unit ??
      (iHistoricMgdl >= 0 || iScanMgdl >= 0 || iStripMgdl >= 0
        ? 'MGDL'
        : iHistoricMmol >= 0 || iScanMmol >= 0 || iStripMmol >= 0
          ? 'MMOLL'
          : null);

    const records: NormalizedRecord[] = [];
    const issues: RowIssue[] = [];
    const warnings: string[] = [];
    if (!unit) warnings.push('Could not determine mg/dL vs mmol/L from LibreView column headers.');

    dataRows.forEach((row, i) => {
      const rowNumber = headerIdx + i + 2;
      const rawRow = row.join(',');
      if (row.every((c) => (c ?? '').trim() === '')) return; // skip blank trailing rows

      const tsCell = iTimestamp >= 0 ? (row[iTimestamp] ?? '') : '';
      const ts = parseTimestamp(tsCell || null, {
        timezone: options.timezone,
        dateOrder: options.dateOrder ?? 'DMY',
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

      const recordType = (iRecordType >= 0 ? (row[iRecordType] ?? '') : '').trim();
      const raw = rawObject(headers, row);
      const notes = iNotes >= 0 ? (row[iNotes] ?? '').trim() : '';

      const glucoseCellFor = (mgdlIdx: number, mmolIdx: number): string => {
        if (mgdlIdx >= 0 && (row[mgdlIdx] ?? '').trim() !== '') return (row[mgdlIdx] ?? '').trim();
        if (mmolIdx >= 0 && (row[mmolIdx] ?? '').trim() !== '') return (row[mmolIdx] ?? '').trim();
        return '';
      };

      let glucoseCell = '';
      let context: GlucoseContext = 'UNKNOWN';
      if (recordType === '0') {
        glucoseCell = glucoseCellFor(iHistoricMgdl, iHistoricMmol);
        context = 'RANDOM';
      } else if (recordType === '1') {
        glucoseCell = glucoseCellFor(iScanMgdl, iScanMmol);
        context = 'RANDOM';
      } else if (recordType === '5') {
        glucoseCell = glucoseCellFor(iStripMgdl, iStripMmol);
      }

      let producedSomething = false;

      if ((recordType === '0' || recordType === '1' || recordType === '5') && glucoseCell !== '') {
        const normalized = normalizeGlucoseValue(Number(glucoseCell), unit);
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
            context: notes ? mapGlucoseContext(notes) : context,
            note: notes || undefined,
            externalId: undefined,
            raw,
          });
          producedSomething = true;
        }
      }

      if (recordType === '3' && iCarbs >= 0) {
        const carbsRaw = (row[iCarbs] ?? '').trim();
        if (carbsRaw !== '') {
          const carbsG = Number(carbsRaw);
          records.push({
            kind: 'meal',
            takenAt: ts.date,
            description: notes || 'Imported meal',
            mealType: 'OTHER',
            carbsG: Number.isFinite(carbsG) ? carbsG : null,
            raw,
          });
          producedSomething = true;
        }
      }

      if (recordType === '2' || recordType === '4') {
        const rapid = iRapidInsulin >= 0 ? (row[iRapidInsulin] ?? '').trim() : '';
        const long = iLongInsulin >= 0 ? (row[iLongInsulin] ?? '').trim() : '';
        if (rapid !== '') {
          records.push({
            kind: 'medication',
            takenAt: ts.date,
            name: 'Rapid-acting insulin',
            dose: rapid,
            raw,
          });
          producedSomething = true;
        }
        if (long !== '') {
          records.push({
            kind: 'medication',
            takenAt: ts.date,
            name: 'Long-acting insulin',
            dose: long,
            raw,
          });
          producedSomething = true;
        }
        if (recordType === '4' && notes) {
          records.push({ kind: 'note', takenAt: ts.date, text: notes, raw });
          producedSomething = true;
        }
      }

      if (!producedSomething) {
        issues.push({
          rowNumber,
          code: 'UNSUPPORTED_ROW',
          message: `Row (Record Type "${recordType}") had no recognised value in a column this connector understands.`,
          rawRow,
        });
      }
    });

    return {
      records,
      issues,
      rowsTotal: dataRows.filter((r) => !r.every((c) => (c ?? '').trim() === '')).length,
      detectedUnit: unit,
      warnings,
    };
  },
};
