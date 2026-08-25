/**
 * Generic, structure-driven XML connector.
 *
 * Expected shape (DiaLog's own documented XML shape, parsed via
 * fast-xml-parser with attributes preserved as `@_name`):
 *
 *   <records>
 *     <record type="glucose" takenAt="2026-01-10T08:10:00Z" value="168" unit="mg/dL" context="fasting" />
 *     <record type="meal" takenAt="..." description="Oatmeal" carbsG="45" />
 *     ...
 *   </records>
 *
 * This is not any particular vendor's XML — Apple Health's export.xml has
 * its own dedicated connector (apple-health.ts) because its shape and
 * identifiers are fixed by Apple.
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
  return typeof v === 'object' && v !== null;
}

function attr(obj: Record<string, unknown>, name: string): string | undefined {
  const v = obj[`@_${name}`];
  if (typeof v === 'string') return v;
  if (typeof v === 'number') return String(v);
  return undefined;
}

function num(v: string | undefined): number | undefined {
  if (v === undefined || v.trim() === '') return undefined;
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
}

/** Finds the array of <record> elements regardless of the exact wrapper element name. */
function findRecordElements(xml: unknown): Record<string, unknown>[] | null {
  if (!isRecord(xml)) return null;
  for (const rootKey of Object.keys(xml)) {
    const root = xml[rootKey];
    if (!isRecord(root)) continue;
    const rec = root.record;
    if (Array.isArray(rec)) return rec.filter(isRecord);
    if (isRecord(rec)) return [rec];
  }
  return null;
}

export const genericXmlConnector: DeviceConnector = {
  id: 'generic-xml',
  name: 'Generic XML',
  vendor: 'Generic',
  description:
    'Structure-driven XML import for DiaLog\'s documented shape: a <records> root containing <record type="..." takenAt="..." .../> elements with kind-specific attributes.',
  howToExport: [
    'Produce an XML file matching the documented DiaLog import shape (see docs/DEVICE_INTEGRATIONS.md).',
    'Upload the XML file here.',
  ],
  acceptedExtensions: ['.xml'],
  kind: 'OTHER',

  detect(sample: ParsedFile): number {
    if (sample.xml === undefined) return 0;
    const records = findRecordElements(sample.xml);
    if (!records || records.length === 0) return 0;
    const first = records[0];
    if (!first || attr(first, 'type') === undefined) return 0;
    return 0.5;
  },

  parse(file: ParsedFile, options: ParseOptions): ParseResult {
    const records = findRecordElements(file.xml);
    if (!records) {
      return {
        records: [],
        issues: [],
        rowsTotal: 0,
        detectedUnit: null,
        warnings: ['No <record> elements found in this XML file.'],
      };
    }

    const glucoseValues: number[] = [];
    for (const el of records) {
      if (attr(el, 'type') === 'glucose') {
        const v = num(attr(el, 'value'));
        if (v !== undefined) glucoseValues.push(v);
      }
    }
    const detectedUnit: GlucoseUnit | null = options.unit ?? inferGlucoseUnit(glucoseValues);

    const out: NormalizedRecord[] = [];
    const issues: RowIssue[] = [];

    records.forEach((el, i) => {
      const rowNumber = i + 1;
      const rawRow = JSON.stringify(el).slice(0, 500);
      const type = attr(el, 'type');
      const ts = parseTimestamp(attr(el, 'takenAt') ?? null, {
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
      const raw = el;

      switch (type) {
        case 'glucose': {
          let unit = detectedUnit;
          const unitStr = attr(el, 'unit')?.toLowerCase();
          if (unitStr?.includes('mmol')) unit = 'MMOLL';
          else if (unitStr?.includes('mg')) unit = 'MGDL';
          const normalized = normalizeGlucoseValue(num(attr(el, 'value')), unit);
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
            takenAt: ts.date,
            valueMgdl: normalized.valueMgdl,
            context: mapGlucoseContext(attr(el, 'context')),
            note: attr(el, 'note'),
            externalId: attr(el, 'id'),
            raw,
          });
          return;
        }
        case 'meal': {
          out.push({
            kind: 'meal',
            takenAt: ts.date,
            description: attr(el, 'description') ?? 'Imported meal',
            mealType: toMealType(attr(el, 'mealType')),
            carbsG: num(attr(el, 'carbsG')) ?? null,
            proteinG: num(attr(el, 'proteinG')) ?? null,
            fatG: num(attr(el, 'fatG')) ?? null,
            fiberG: num(attr(el, 'fiberG')) ?? null,
            calories: num(attr(el, 'calories')) ?? null,
            externalId: attr(el, 'id'),
            raw,
          });
          return;
        }
        case 'medication': {
          const name = attr(el, 'name');
          if (!name) {
            issues.push({
              rowNumber,
              code: 'MISSING_VALUE',
              message: 'medication record has no name attribute.',
              rawRow,
            });
            return;
          }
          out.push({
            kind: 'medication',
            takenAt: ts.date,
            name,
            dose: attr(el, 'dose') ?? null,
            externalId: attr(el, 'id'),
            raw,
          });
          return;
        }
        case 'weight': {
          const weightKg = num(attr(el, 'weightKg') ?? attr(el, 'value'));
          if (weightKg === undefined) {
            issues.push({
              rowNumber,
              code: 'MISSING_VALUE',
              message: 'weight record has no weightKg attribute.',
              rawRow,
            });
            return;
          }
          out.push({ kind: 'weight', takenAt: ts.date, weightKg, externalId: attr(el, 'id'), raw });
          return;
        }
        case 'note': {
          const text = attr(el, 'text');
          if (!text) {
            issues.push({
              rowNumber,
              code: 'MISSING_VALUE',
              message: 'note record has no text attribute.',
              rawRow,
            });
            return;
          }
          out.push({ kind: 'note', takenAt: ts.date, text, externalId: attr(el, 'id'), raw });
          return;
        }
        case 'exercise': {
          const activity = attr(el, 'activity');
          const durationMin = num(attr(el, 'durationMin'));
          if (!activity || durationMin === undefined) {
            issues.push({
              rowNumber,
              code: 'MISSING_VALUE',
              message: 'exercise record needs activity and durationMin attributes.',
              rawRow,
            });
            return;
          }
          out.push({
            kind: 'exercise',
            takenAt: ts.date,
            activity,
            durationMin,
            distanceKm: num(attr(el, 'distanceKm')) ?? null,
            steps: num(attr(el, 'steps')) ?? null,
            externalId: attr(el, 'id'),
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

    return { records: out, issues, rowsTotal: records.length, detectedUnit, warnings: [] };
  },
};
