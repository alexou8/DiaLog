/**
 * Nightscout `entries.json` connector.
 *
 * Nightscout (the open-source "#WeAreNotWaiting" CGM remote-monitoring
 * project; nightscout.github.io) exposes and can export its `entries`
 * collection as JSON, documented at
 * https://github.com/nightscout/documentation (see docs/DEVICE_INTEGRATIONS.md
 * for the exact source and date checked). Each entry is an object such as:
 *
 *   { "_id": "...", "type": "sgv", "sgv": 145, "date": 1700000000000,
 *     "dateString": "2023-11-14T22:13:20.000Z", "direction": "Flat",
 *     "device": "..." }
 *
 * `type: "sgv"` is a CGM sensor glucose value (always mg/dL); `type: "mbg"`
 * is a meter blood glucose calibration reading, using the `mbg` field
 * instead of `sgv`. `date` is epoch milliseconds; `dateString` is an ISO
 * string — this connector prefers `dateString` when present, falling back
 * to `date`.
 */
import { normalizeGlucoseValue } from '../normalize';
import { parseTimestamp } from '../parse';
import type {
  DeviceConnector,
  NormalizedRecord,
  ParseOptions,
  ParseResult,
  ParsedFile,
  RowIssue,
} from '../types';

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

function asArray(json: unknown): unknown[] | null {
  return Array.isArray(json) ? json : null;
}

export const nightscoutConnector: DeviceConnector = {
  id: 'nightscout',
  name: 'Nightscout entries.json',
  vendor: 'Nightscout (open source)',
  description:
    'Reads a Nightscout `entries.json` export (sgv = CGM sensor glucose, mbg = meter calibration reading), always in mg/dL per the Nightscout API.',
  howToExport: [
    "From your Nightscout site, request GET /api/v1/entries.json?count=<n> (add an API token/secret if your site requires one), or use your Nightscout admin tools' export feature.",
    'Save the JSON response to a file.',
    'Upload that JSON file here.',
  ],
  acceptedExtensions: ['.json'],
  kind: 'CGM',

  detect(sample: ParsedFile): number {
    const arr = asArray(sample.json);
    if (!arr || arr.length === 0) return 0;
    const first = arr[0];
    if (!isRecord(first)) return 0;
    const hasType = first.type === 'sgv' || first.type === 'mbg';
    const hasGlucoseField = 'sgv' in first || 'mbg' in first;
    return hasType && hasGlucoseField ? 0.85 : 0;
  },

  parse(file: ParsedFile, options: ParseOptions): ParseResult {
    const arr = asArray(file.json);
    if (!arr) {
      return {
        records: [],
        issues: [],
        rowsTotal: 0,
        detectedUnit: null,
        warnings: ['JSON file was not an array of Nightscout entries.'],
      };
    }

    const records: NormalizedRecord[] = [];
    const issues: RowIssue[] = [];

    arr.forEach((item, i) => {
      const rowNumber = i + 1;
      if (!isRecord(item)) {
        issues.push({
          rowNumber,
          code: 'UNSUPPORTED_ROW',
          message: 'Array element is not an object.',
        });
        return;
      }
      const rawRow = JSON.stringify(item).slice(0, 500);
      const type = item.type;
      if (type !== 'sgv' && type !== 'mbg') {
        issues.push({
          rowNumber,
          code: 'UNSUPPORTED_ROW',
          message: `Unrecognised entry type "${String(type)}" (expected "sgv" or "mbg").`,
          rawRow,
        });
        return;
      }

      const dateString = typeof item.dateString === 'string' ? item.dateString : undefined;
      const dateMs = typeof item.date === 'number' ? item.date : undefined;
      const ts = parseTimestamp(dateString ?? dateMs ?? null, {
        timezone: options.timezone,
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

      const value = type === 'sgv' ? item.sgv : item.mbg;
      const num =
        typeof value === 'number' ? value : typeof value === 'string' ? Number(value) : NaN;
      const normalized = normalizeGlucoseValue(num, 'MGDL');
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
        context: type === 'mbg' ? 'RANDOM' : 'UNKNOWN',
        note: typeof item.direction === 'string' ? `Trend: ${item.direction}` : undefined,
        externalId: typeof item._id === 'string' ? item._id : undefined,
        raw: item,
      });
    });

    return { records, issues, rowsTotal: arr.length, detectedUnit: 'MGDL', warnings: [] };
  },
};
