/**
 * Ingestion framework types.
 *
 * A `DeviceConnector` turns a low-level `ParsedFile` (already parsed out of
 * CSV/XLSX/XML/JSON text into rows or structured data) into `NormalizedRecord`s
 * the caller can persist, plus `RowIssue`s for anything that could not be
 * imported. Connectors never touch the database and never silently drop or
 * mutate a row: every row that isn't turned into a record must produce an
 * issue, so nothing disappears without explanation.
 */
import type { GlucoseContext, GlucoseUnit, MealType } from '@prisma/client';

export type { GlucoseContext, GlucoseUnit, MealType };

export interface BaseRecord {
  /** Timestamp the event actually occurred, in UTC. */
  takenAt: Date;
  /** The untouched source row/object, kept for provenance (ImportIssue-free rows still keep raw payload). */
  raw: Record<string, unknown>;
}

export interface GlucoseRecord extends BaseRecord {
  kind: 'glucose';
  valueMgdl: number;
  context: GlucoseContext;
  note?: string;
  externalId?: string;
}

export interface MealRecord extends BaseRecord {
  kind: 'meal';
  description: string;
  mealType: MealType;
  carbsG?: number | null;
  proteinG?: number | null;
  fatG?: number | null;
  fiberG?: number | null;
  calories?: number | null;
  externalId?: string;
}

export interface MedicationRecord extends BaseRecord {
  kind: 'medication';
  name: string;
  dose?: string | null;
  route?: string | null;
  externalId?: string;
}

export interface ExerciseRecord extends BaseRecord {
  kind: 'exercise';
  activity: string;
  durationMin: number;
  intensity?: 'LIGHT' | 'MODERATE' | 'VIGOROUS';
  distanceKm?: number | null;
  steps?: number | null;
  externalId?: string;
}

export interface SleepRecord extends BaseRecord {
  kind: 'sleep';
  endedAt: Date;
  durationMin: number;
  quality?: number | null;
  externalId?: string;
}

export interface WeightRecord extends BaseRecord {
  kind: 'weight';
  weightKg: number;
  externalId?: string;
}

export interface NoteRecord extends BaseRecord {
  kind: 'note';
  text: string;
  externalId?: string;
}

export type NormalizedRecord =
  | GlucoseRecord
  | MealRecord
  | MedicationRecord
  | ExerciseRecord
  | SleepRecord
  | WeightRecord
  | NoteRecord;

export type RowIssueCode =
  | 'MISSING_TIMESTAMP'
  | 'INVALID_TIMESTAMP'
  | 'INVALID_VALUE'
  | 'OUT_OF_RANGE'
  | 'UNKNOWN_UNIT'
  | 'MISSING_VALUE'
  | 'UNSUPPORTED_ROW'
  | 'FUTURE_TIMESTAMP'
  | 'PARSE_ERROR';

export interface RowIssue {
  rowNumber: number;
  code: RowIssueCode;
  message: string;
  rawRow?: string;
}

export interface ParseResult {
  records: NormalizedRecord[];
  issues: RowIssue[];
  rowsTotal: number;
  detectedUnit: GlucoseUnit | null;
  warnings: string[];
}

/** Low-level parsed representation of an uploaded file, before connector-specific interpretation. */
export interface ParsedFile {
  filename: string;
  mimeType: string;
  /** Raw decoded text, when the source is text-based (CSV/XML/JSON) or plain text. */
  text?: string;
  /** Row-major grid of string cells, from CSV or XLSX. rows[0] is typically the header row. */
  rows?: string[][];
  /** Parsed JSON value, when the source is JSON. */
  json?: unknown;
  /** Parsed XML value (via fast-xml-parser), when the source is XML. */
  xml?: unknown;
}

export type DateOrder = 'DMY' | 'MDY';

export interface ParseOptions {
  /** IANA timezone used to interpret timestamps that carry no offset. Defaults to UTC. */
  timezone?: string;
  /** Force day/month vs month/day interpretation for ambiguous dates (e.g. "01/02/2026"). */
  dateOrder?: DateOrder;
  /** Explicit glucose unit override, bypassing auto-detection. */
  unit?: GlucoseUnit;
  /** Reference "now" for future-timestamp rejection; defaults to `new Date()`. Exposed for tests. */
  now?: Date;
}

export interface DeviceConnector {
  /** Stable machine identifier, e.g. "abbott-libreview". Persisted as ImportBatch.connectorId. */
  id: string;
  /** Human-facing name, e.g. "Abbott LibreView (FreeStyle Libre)". */
  name: string;
  vendor: string;
  description: string;
  /** Step-by-step instructions for how the user produces this export file. */
  howToExport: string[];
  acceptedExtensions: string[];
  kind:
    | 'GLUCOSE_METER'
    | 'CGM'
    | 'BLOOD_PRESSURE_MONITOR'
    | 'SCALE'
    | 'WEARABLE'
    | 'PHONE_HEALTH_PLATFORM'
    | 'OTHER';
  /** Confidence in [0, 1] that this connector understands `sample`. 0 = definitely not. */
  detect(sample: ParsedFile): number;
  parse(file: ParsedFile, options: ParseOptions): Promise<ParseResult> | ParseResult;
}
