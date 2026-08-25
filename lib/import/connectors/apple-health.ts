/**
 * Apple Health `export.xml` connector.
 *
 * Apple's Health app export produces a `export.xml` (inside the
 * `export.zip` you get from Health > profile icon > Export All Health
 * Data). This file's schema is not formally documented by Apple, but its
 * shape is stable and widely reverse-engineered: a flat list of
 * `<Record type="HKQuantityTypeIdentifier..." sourceName="..."
 * unit="..." startDate="..." endDate="..." value="..." .../>` elements
 * (see docs/DEVICE_INTEGRATIONS.md for sources checked).
 *
 * We only pull the identifiers relevant to DiaLog:
 *   - HKQuantityTypeIdentifierBloodGlucose      -> glucose reading
 *   - HKQuantityTypeIdentifierDietaryCarbohydrates -> meal (carbs only)
 *   - HKQuantityTypeIdentifierBodyMass          -> weight
 *   - HKCategoryTypeIdentifierSleepAnalysis     -> sleep session
 *   - HKWorkoutType / <Workout .../> elements   -> exercise session
 *
 * `startDate`/`endDate` use the format `YYYY-MM-DD HH:mm:ss ±HHMM`, which
 * `parseTimestamp`'s ISO-with-space branch does not directly cover (no
 * colon in the offset) — this connector normalises the offset to `±HH:MM`
 * before delegating to `parseTimestamp`.
 *
 * export.xml can be very large (hundreds of MB for a multi-year export);
 * this connector relies on parse.ts's MAX_FILE_BYTES guard rather than
 * attempting true streaming XML parsing, and only reads `<Record>` and
 * `<Workout>` elements out of the parsed tree, dropping everything else
 * (correlations, clinical records, activity summaries) rather than
 * misinterpreting them.
 */
import { mapGlucoseContext, normalizeGlucoseValue } from '../normalize';
import type {
  DeviceConnector,
  GlucoseUnit,
  NormalizedRecord,
  ParseOptions,
  ParseResult,
  ParsedFile,
  RowIssue,
} from '../types';

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null;
}

function attr(obj: Record<string, unknown>, name: string): string | undefined {
  const v = obj[`@_${name}`];
  if (typeof v === 'string') return v;
  if (typeof v === 'number') return String(v);
  return undefined;
}

function asElementArray(v: unknown): Record<string, unknown>[] {
  if (Array.isArray(v)) return v.filter(isRecord);
  if (isRecord(v)) return [v];
  return [];
}

/** Apple's date format is `YYYY-MM-DD HH:mm:ss ±HHMM`; normalise the offset to `±HH:MM` for parseTimestamp. */
function parseAppleDate(v: string | undefined): Date | null {
  if (!v) return null;
  const m = /^(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}) ([+-]\d{2})(\d{2})$/.exec(v.trim());
  if (!m) {
    const d = new Date(v);
    return Number.isNaN(d.getTime()) ? null : d;
  }
  const [, datePart, timePart, offH, offM] = m;
  const iso = `${datePart}T${timePart}${offH}:${offM}`;
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? null : d;
}

const GLUCOSE_TYPE = 'HKQuantityTypeIdentifierBloodGlucose';
const CARBS_TYPE = 'HKQuantityTypeIdentifierDietaryCarbohydrates';
const WEIGHT_TYPE = 'HKQuantityTypeIdentifierBodyMass';
const SLEEP_TYPE = 'HKCategoryTypeIdentifierSleepAnalysis';

function findHealthDataRoot(xml: unknown): Record<string, unknown> | null {
  if (!isRecord(xml)) return null;
  const hd = xml.HealthData;
  return isRecord(hd) ? hd : null;
}

function rawObject(el: Record<string, unknown>): Record<string, unknown> {
  return el;
}

export const appleHealthConnector: DeviceConnector = {
  id: 'apple-health',
  name: 'Apple Health export',
  vendor: 'Apple',
  description:
    "Reads Apple Health's export.xml (Health app > profile icon > Export All Health Data > export.zip), pulling blood glucose, dietary carbohydrates, body mass, sleep analysis, and workout records. Schema is not formally published by Apple; layout confirmed against widely-documented community references.",
  howToExport: [
    'Open the Health app on iPhone.',
    'Tap your profile picture, then "Export All Health Data".',
    'Unzip the downloaded export.zip on a computer.',
    'Upload the export.xml file here (large files may take a while).',
  ],
  acceptedExtensions: ['.xml'],
  kind: 'PHONE_HEALTH_PLATFORM',

  detect(sample: ParsedFile): number {
    const root = findHealthDataRoot(sample.xml);
    if (!root) return 0;
    const hasRecords = 'Record' in root || 'Workout' in root;
    return hasRecords ? 0.85 : 0;
  },

  parse(file: ParsedFile, options: ParseOptions): ParseResult {
    const root = findHealthDataRoot(file.xml);
    if (!root) {
      return {
        records: [],
        issues: [],
        rowsTotal: 0,
        detectedUnit: null,
        warnings: ['No <HealthData> root with <Record> elements found.'],
      };
    }
    const recordEls = asElementArray(root.Record);
    const workoutEls = asElementArray(root.Workout);
    const relevantTypes = new Set([GLUCOSE_TYPE, CARBS_TYPE, WEIGHT_TYPE, SLEEP_TYPE]);
    const relevantRecordCount = recordEls.filter((el) =>
      relevantTypes.has(attr(el, 'type') ?? ''),
    ).length;

    const glucoseValues: number[] = [];
    for (const el of recordEls) {
      if (attr(el, 'type') === GLUCOSE_TYPE) {
        const v = Number(attr(el, 'value'));
        if (Number.isFinite(v)) glucoseValues.push(v);
      }
    }
    const detectedUnit: GlucoseUnit | null =
      options.unit ??
      (glucoseValues.length > 0
        ? recordEls.find(
            (e) =>
              attr(e, 'type') === GLUCOSE_TYPE && attr(e, 'unit')?.toLowerCase().includes('mmol'),
          )
          ? 'MMOLL'
          : 'MGDL'
        : null);

    const out: NormalizedRecord[] = [];
    const issues: RowIssue[] = [];

    recordEls.forEach((el, i) => {
      const rowNumber = i + 1;
      const type = attr(el, 'type');
      if (
        type !== GLUCOSE_TYPE &&
        type !== CARBS_TYPE &&
        type !== WEIGHT_TYPE &&
        type !== SLEEP_TYPE
      )
        return;

      const rawRow = JSON.stringify(el).slice(0, 500);
      const start = parseAppleDate(attr(el, 'startDate'));
      if (!start) {
        issues.push({
          rowNumber,
          code: 'INVALID_TIMESTAMP',
          message: `Could not parse startDate "${attr(el, 'startDate') ?? ''}".`,
          rawRow,
        });
        return;
      }
      const maxFuture = (options.now ?? new Date()).getTime() + 24 * 3600 * 1000;
      if (start.getTime() > maxFuture) {
        issues.push({
          rowNumber,
          code: 'FUTURE_TIMESTAMP',
          message: `startDate ${start.toISOString()} is more than 24 hours in the future.`,
          rawRow,
        });
        return;
      }

      if (type === GLUCOSE_TYPE) {
        const unitStr = attr(el, 'unit')?.toLowerCase();
        const unit: GlucoseUnit | null = unitStr?.includes('mmol')
          ? 'MMOLL'
          : unitStr?.includes('mg')
            ? 'MGDL'
            : detectedUnit;
        const normalized = normalizeGlucoseValue(Number(attr(el, 'value')), unit);
        if (normalized.valueMgdl === null) {
          issues.push({
            rowNumber,
            code: normalized.issue ?? 'INVALID_VALUE',
            message: normalized.message ?? 'Invalid glucose value.',
            rawRow,
          });
          return;
        }
        out.push({
          kind: 'glucose',
          takenAt: start,
          valueMgdl: normalized.valueMgdl,
          context: mapGlucoseContext(attr(el, 'metadata')),
          externalId: undefined,
          raw: rawObject(el),
        });
        return;
      }

      if (type === CARBS_TYPE) {
        const carbsG = Number(attr(el, 'value'));
        if (!Number.isFinite(carbsG)) {
          issues.push({
            rowNumber,
            code: 'INVALID_VALUE',
            message: 'Dietary carbohydrates value is not a number.',
            rawRow,
          });
          return;
        }
        out.push({
          kind: 'meal',
          takenAt: start,
          description: 'Imported meal (Apple Health)',
          mealType: 'OTHER',
          carbsG,
          raw: rawObject(el),
        });
        return;
      }

      if (type === WEIGHT_TYPE) {
        const rawVal = Number(attr(el, 'value'));
        if (!Number.isFinite(rawVal)) {
          issues.push({
            rowNumber,
            code: 'INVALID_VALUE',
            message: 'Body mass value is not a number.',
            rawRow,
          });
          return;
        }
        const unitStr = attr(el, 'unit')?.toLowerCase();
        const weightKg = unitStr?.includes('lb') ? rawVal * 0.45359237 : rawVal;
        out.push({ kind: 'weight', takenAt: start, weightKg, raw: rawObject(el) });
        return;
      }

      if (type === SLEEP_TYPE) {
        const end = parseAppleDate(attr(el, 'endDate'));
        if (!end) {
          issues.push({
            rowNumber,
            code: 'INVALID_TIMESTAMP',
            message: `Could not parse endDate "${attr(el, 'endDate') ?? ''}".`,
            rawRow,
          });
          return;
        }
        const durationMin = Math.round((end.getTime() - start.getTime()) / 60000);
        if (durationMin <= 0) {
          issues.push({
            rowNumber,
            code: 'INVALID_VALUE',
            message: 'Sleep record endDate is not after startDate.',
            rawRow,
          });
          return;
        }
        out.push({ kind: 'sleep', takenAt: start, endedAt: end, durationMin, raw: rawObject(el) });
      }
    });

    workoutEls.forEach((el, i) => {
      const rowNumber = recordEls.length + i + 1;
      const rawRow = JSON.stringify(el).slice(0, 500);
      const start = parseAppleDate(attr(el, 'startDate'));
      if (!start) {
        issues.push({
          rowNumber,
          code: 'INVALID_TIMESTAMP',
          message: `Could not parse workout startDate "${attr(el, 'startDate') ?? ''}".`,
          rawRow,
        });
        return;
      }
      const maxFuture = (options.now ?? new Date()).getTime() + 24 * 3600 * 1000;
      if (start.getTime() > maxFuture) {
        issues.push({
          rowNumber,
          code: 'FUTURE_TIMESTAMP',
          message: `startDate ${start.toISOString()} is more than 24 hours in the future.`,
          rawRow,
        });
        return;
      }
      const end = parseAppleDate(attr(el, 'endDate'));
      const durationRaw = Number(attr(el, 'duration'));
      const durationMin = Number.isFinite(durationRaw)
        ? Math.round(durationRaw)
        : end
          ? Math.round((end.getTime() - start.getTime()) / 60000)
          : NaN;
      if (!Number.isFinite(durationMin) || durationMin <= 0) {
        issues.push({
          rowNumber,
          code: 'MISSING_VALUE',
          message: 'Workout record has no usable duration.',
          rawRow,
        });
        return;
      }
      out.push({
        kind: 'exercise',
        takenAt: start,
        activity:
          attr(el, 'workoutActivityType')?.replace('HKWorkoutActivityType', '') ?? 'Workout',
        durationMin,
        distanceKm: (() => {
          const d = Number(attr(el, 'totalDistance'));
          if (!Number.isFinite(d)) return null;
          const unit = attr(el, 'totalDistanceUnit')?.toLowerCase();
          return unit?.includes('mi') ? d * 1.609344 : d;
        })(),
        raw: rawObject(el),
      });
    });

    return {
      records: out,
      issues,
      rowsTotal: relevantRecordCount + workoutEls.length,
      detectedUnit,
      warnings: [],
    };
  },
};
