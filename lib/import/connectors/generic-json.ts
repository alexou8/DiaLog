/**
 * Generic, structure-driven JSON connector.
 *
 * Expected shape: either a bare array, or an object with a `records` (or
 * `entries` / `data`) array property, of objects shaped like:
 *
 *   {
 *     "type": "glucose" | "meal" | "medication" | "exercise" | "sleep" | "weight" | "note",
 *     "takenAt": "2026-01-10T08:10:00Z",   // or "timestamp" / "date"
 *     "value": 168,                          // glucose mg/dL or mmol/L
 *     "unit": "mg/dL" | "mmol/L",            // optional, glucose only
 *     "context": "fasting",                  // optional, glucose only, free text
 *     "description" / "name" / "activity" / "text": "...",
 *     "carbsG", "proteinG", "fatG", "fiberG", "calories": number,
 *     "dose", "durationMin", "distanceKm", "steps", "weightKg", "endedAt", "quality": ...
 *   }
 *
 * This is DiaLog's own documented shape for hand-written or scripted JSON
 * imports — it is not any particular vendor's format. Unrecognised objects
 * produce an UNSUPPORTED_ROW issue rather than being guessed at.
 */
import { mapGlucoseContext, normalizeGlucoseValue } from '../normalize';
import { parseTimestamp } from '../parse';
import { inferGlucoseUnit } from '@/lib/domain/units';
import type {
  DeviceConnector,
  GlucoseUnit,
  MealType,
  NormalizedRecord,
  ParseOptions,
  ParseResult,
  ParsedFile,
  RowIssue,
} from '../types';

const MEAL_TYPES: readonly MealType[] = ['BREAKFAST', 'LUNCH', 'DINNER', 'SNACK', 'OTHER'];

function toMealType(v: string | undefined): MealType {
  const upper = v?.toUpperCase();
  return (MEAL_TYPES as readonly string[]).includes(upper ?? '') ? (upper as MealType) : 'OTHER';
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

function asArray(json: unknown): unknown[] | null {
  if (Array.isArray(json)) return json;
  if (isRecord(json)) {
    for (const key of ['records', 'entries', 'data']) {
      const v = json[key];
      if (Array.isArray(v)) return v;
    }
  }
  return null;
}

function str(v: unknown): string | undefined {
  if (typeof v === 'string') return v;
  if (typeof v === 'number') return String(v);
  return undefined;
}

function num(v: unknown): number | undefined {
  if (typeof v === 'number' && Number.isFinite(v)) return v;
  if (typeof v === 'string' && v.trim() !== '' && Number.isFinite(Number(v))) return Number(v);
  return undefined;
}

function timestampField(obj: Record<string, unknown>): string | number | undefined {
  const v = obj.takenAt ?? obj.timestamp ?? obj.date ?? obj.time;
  if (typeof v === 'string' || typeof v === 'number') return v;
  return undefined;
}

export const genericJsonConnector: DeviceConnector = {
  id: 'generic-json',
  name: 'Generic JSON',
  vendor: 'Generic',
  description:
    "Structure-driven JSON import for DiaLog's documented record shape: an array (or {records:[...]}) of objects with a `type`, a timestamp field, and kind-specific value fields.",
  howToExport: [
    'Produce a JSON file matching the documented DiaLog import shape (see docs/DEVICE_INTEGRATIONS.md), or export from a tool that already writes it.',
    'Upload the JSON file here.',
  ],
  acceptedExtensions: ['.json'],
  kind: 'OTHER',

  detect(sample: ParsedFile): number {
    if (sample.json === undefined) return 0;
    const arr = asArray(sample.json);
    if (!arr || arr.length === 0) return 0;
    const first = arr[0];
    if (!isRecord(first)) return 0;
    if (typeof first.type !== 'string') return 0;
    return 0.55;
  },

  parse(file: ParsedFile, options: ParseOptions): ParseResult {
    const arr = asArray(file.json);
    if (!arr) {
      return {
        records: [],
        issues: [],
        rowsTotal: 0,
        detectedUnit: null,
        warnings: ['JSON file did not contain a records array.'],
      };
    }

    const glucoseValues: number[] = [];
    for (const item of arr) {
      if (isRecord(item) && item.type === 'glucose') {
        const v = num(item.value ?? item.valueMgdl);
        if (v !== undefined) glucoseValues.push(v);
      }
    }
    const detectedUnit: GlucoseUnit | null = options.unit ?? inferGlucoseUnit(glucoseValues);

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
      const raw = item;
      const type = str(item.type);
      const tsField = timestampField(item);
      const ts = parseTimestamp(tsField ?? null, {
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

      switch (type) {
        case 'glucose': {
          const v = num(item.value ?? item.valueMgdl);
          let unit = detectedUnit;
          const unitStr = str(item.unit)?.toLowerCase();
          if (unitStr?.includes('mmol')) unit = 'MMOLL';
          else if (unitStr?.includes('mg')) unit = 'MGDL';
          const normalized = normalizeGlucoseValue(v, unit);
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
            context: mapGlucoseContext(str(item.context)),
            note: str(item.note),
            externalId: str(item.id ?? item.externalId),
            raw,
          });
          return;
        }
        case 'meal': {
          const description = str(item.description) ?? 'Imported meal';
          records.push({
            kind: 'meal',
            takenAt: ts.date,
            description,
            mealType: toMealType(str(item.mealType)),
            carbsG: num(item.carbsG) ?? null,
            proteinG: num(item.proteinG) ?? null,
            fatG: num(item.fatG) ?? null,
            fiberG: num(item.fiberG) ?? null,
            calories: num(item.calories) ?? null,
            externalId: str(item.id ?? item.externalId),
            raw,
          });
          return;
        }
        case 'medication': {
          const name = str(item.name);
          if (!name) {
            issues.push({
              rowNumber,
              code: 'MISSING_VALUE',
              message: 'medication record has no name.',
              rawRow,
            });
            return;
          }
          records.push({
            kind: 'medication',
            takenAt: ts.date,
            name,
            dose: str(item.dose) ?? null,
            externalId: str(item.id ?? item.externalId),
            raw,
          });
          return;
        }
        case 'exercise': {
          const activity = str(item.activity);
          const durationMin = num(item.durationMin);
          if (!activity || durationMin === undefined) {
            issues.push({
              rowNumber,
              code: 'MISSING_VALUE',
              message: 'exercise record needs activity and durationMin.',
              rawRow,
            });
            return;
          }
          records.push({
            kind: 'exercise',
            takenAt: ts.date,
            activity,
            durationMin,
            distanceKm: num(item.distanceKm) ?? null,
            steps: num(item.steps) ?? null,
            externalId: str(item.id ?? item.externalId),
            raw,
          });
          return;
        }
        case 'sleep': {
          const endedField = str(item.endedAt);
          const endedTs = parseTimestamp(endedField ?? null, {
            timezone: options.timezone,
            now: options.now,
          });
          const durationMin = num(item.durationMin);
          if (!endedTs.date || durationMin === undefined) {
            issues.push({
              rowNumber,
              code: 'MISSING_VALUE',
              message: 'sleep record needs endedAt and durationMin.',
              rawRow,
            });
            return;
          }
          records.push({
            kind: 'sleep',
            takenAt: ts.date,
            endedAt: endedTs.date,
            durationMin,
            quality: num(item.quality) ?? null,
            externalId: str(item.id ?? item.externalId),
            raw,
          });
          return;
        }
        case 'weight': {
          const weightKg = num(item.weightKg ?? item.value);
          if (weightKg === undefined) {
            issues.push({
              rowNumber,
              code: 'MISSING_VALUE',
              message: 'weight record has no weightKg.',
              rawRow,
            });
            return;
          }
          records.push({
            kind: 'weight',
            takenAt: ts.date,
            weightKg,
            externalId: str(item.id ?? item.externalId),
            raw,
          });
          return;
        }
        case 'note': {
          const text = str(item.text) ?? str(item.note);
          if (!text) {
            issues.push({
              rowNumber,
              code: 'MISSING_VALUE',
              message: 'note record has no text.',
              rawRow,
            });
            return;
          }
          records.push({
            kind: 'note',
            takenAt: ts.date,
            text,
            externalId: str(item.id ?? item.externalId),
            raw,
          });
          return;
        }
        default:
          issues.push({
            rowNumber,
            code: 'UNSUPPORTED_ROW',
            message: `Unrecognised record type "${String(type)}".`,
            rawRow,
          });
      }
    });

    return { records, issues, rowsTotal: arr.length, detectedUnit, warnings: [] };
  },
};
