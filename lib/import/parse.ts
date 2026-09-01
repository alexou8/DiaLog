/**
 * Low-level file parsing: format detection, RFC4180 CSV, XLSX (via exceljs),
 * XML (via fast-xml-parser), JSON, and timestamp parsing shared by every
 * connector.
 *
 * This module never talks to a database and never assumes a particular
 * connector's column layout — that interpretation lives in
 * `lib/import/connectors/*`.
 */
import ExcelJS from 'exceljs';
import { XMLParser } from 'fast-xml-parser';
import type { DateOrder, ParsedFile } from './types';

// ------------------------------------------------------------- size guards

/** Hard ceiling on any file this module will attempt to parse in memory. */
export const MAX_FILE_BYTES = 100 * 1024 * 1024; // 100 MB

/** Ceiling specifically for JSON.parse, which is O(n) memory on top of the string. */
export const MAX_JSON_BYTES = 50 * 1024 * 1024; // 50 MB

// ------------------------------------------------------------------- BOM

export function stripBom(text: string): string {
  if (text.length > 0 && text.charCodeAt(0) === 0xfeff) return text.slice(1);
  return text;
}

// ------------------------------------------------------------ delimiter

const CANDIDATE_DELIMITERS = [',', ';', '\t', '|'] as const;

/**
 * Auto-detects the CSV delimiter by counting occurrences of each candidate
 * outside of quoted spans across the first few lines and picking the most
 * consistent, most frequent one. Defaults to comma.
 */
export function detectDelimiter(text: string): string {
  const sampleLines = text
    .split(/\r\n|\r|\n/)
    .slice(0, 10)
    .filter((l) => l.length > 0);
  if (sampleLines.length === 0) return ',';

  let best: string = ',';
  let bestScore = -1;
  for (const delim of CANDIDATE_DELIMITERS) {
    const counts = sampleLines.map((line) => countOutsideQuotes(line, delim));
    const total = counts.reduce((a, b) => a + b, 0);
    if (total === 0) continue;
    // Prefer delimiters whose per-line count is consistent (low variance) and non-zero.
    const first = counts[0] ?? 0;
    const consistent = counts.every((c) => c === first);
    const score = total + (consistent ? 1000 : 0);
    if (score > bestScore) {
      bestScore = score;
      best = delim;
    }
  }
  return best;
}

function countOutsideQuotes(line: string, ch: string): number {
  let count = 0;
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (c === '"') {
      inQuotes = !inQuotes;
    } else if (!inQuotes && c === ch) {
      count++;
    }
  }
  return count;
}

// ---------------------------------------------------------------- CSV

/**
 * A hand-written RFC4180 CSV parser: handles quoted fields, embedded commas,
 * embedded newlines inside quotes, escaped quotes (""), BOM stripping, and
 * auto delimiter detection. Returns a row-major grid of string cells; ragged
 * rows are preserved as-is (short rows simply have fewer cells) so downstream
 * connectors can report MISSING_VALUE issues rather than silently misaligning
 * columns.
 */
export function parseCsv(input: string, delimiter?: string): string[][] {
  const text = stripBom(input);
  const delim = delimiter ?? detectDelimiter(text);
  const rows: string[][] = [];
  let row: string[] = [];
  let field = '';
  let inQuotes = false;
  let i = 0;
  const n = text.length;
  let sawAnyField = false;

  const pushField = () => {
    row.push(field);
    field = '';
  };
  const pushRow = () => {
    pushField();
    rows.push(row);
    row = [];
    sawAnyField = false;
  };

  while (i < n) {
    const c = text[i] as string;
    if (inQuotes) {
      if (c === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i += 2;
          continue;
        }
        inQuotes = false;
        i++;
        continue;
      }
      field += c;
      i++;
      continue;
    }
    if (c === '"' && field.length === 0) {
      inQuotes = true;
      sawAnyField = true;
      i++;
      continue;
    }
    if (c === delim) {
      sawAnyField = true;
      pushField();
      i++;
      continue;
    }
    if (c === '\r') {
      if (text[i + 1] === '\n') i++;
      if (sawAnyField || field.length > 0) pushRow();
      i++;
      continue;
    }
    if (c === '\n') {
      if (sawAnyField || field.length > 0) pushRow();
      i++;
      continue;
    }
    field += c;
    sawAnyField = true;
    i++;
  }
  // Flush trailing field/row if the file doesn't end with a newline.
  if (sawAnyField || field.length > 0 || row.length > 0) {
    pushRow();
  }

  return rows;
}

// --------------------------------------------------------------- XLSX

/** Parses an XLSX workbook's first worksheet into a row-major grid of strings. */
export async function parseXlsx(buffer: ArrayBuffer | Buffer): Promise<string[][]> {
  const workbook = new ExcelJS.Workbook();
  await workbook.xlsx.load(buffer as ExcelJS.Buffer);
  const sheet = workbook.worksheets[0];
  if (!sheet) return [];
  const rows: string[][] = [];
  sheet.eachRow({ includeEmpty: false }, (row) => {
    const cells: string[] = [];
    // ExcelJS rows are 1-indexed and sparse; walk to the declared cell count.
    for (let c = 1; c <= row.cellCount; c++) {
      const cell = row.getCell(c);
      cells.push(cellToString(cell.value));
    }
    rows.push(cells);
  });
  return rows;
}

function cellToString(value: ExcelJS.CellValue): string {
  if (value === null || value === undefined) return '';
  if (value instanceof Date) return value.toISOString();
  if (typeof value === 'object') {
    if ('text' in value && typeof value.text === 'string') return value.text;
    if ('result' in value) return cellToString((value as { result: ExcelJS.CellValue }).result);
    if ('richText' in value && Array.isArray(value.richText)) {
      return value.richText.map((t) => t.text).join('');
    }
    return String(value);
  }
  return String(value);
}

// ---------------------------------------------------------------- XML

/**
 * XML entity expansion ("billion laughs") is a decompression-bomb attack: a
 * few kilobytes of nested DOCTYPE entity definitions expand to gigabytes
 * during parsing, so the MAX_FILE_BYTES ceiling below does not bound it — the
 * blow-up happens after the size check passes. fast-xml-parser has carried a
 * series of advisories for exactly this (GHSA-jmr7-xgp7-cmfj and its
 * incomplete fixes), so the pinned version is only half the defence.
 *
 * None of the formats DiaLog imports uses a DTD: Apple Health's export.xml,
 * LibreView and the generic-XML fallback are all plain element trees. So we
 * reject any internal subset outright rather than relying on the parser's
 * expansion limits. This keeps standard predefined entities (`&amp;`, which
 * does appear in Apple Health `sourceName` attributes) working, which
 * `processEntities: false` would have broken.
 */
const DOCTYPE_WITH_INTERNAL_SUBSET = /<!DOCTYPE[^>[]*\[/i;

export function parseXmlText(text: string): unknown {
  if (DOCTYPE_WITH_INTERNAL_SUBSET.test(text)) {
    throw new Error(
      'XML file declares a DOCTYPE internal subset (custom entities), which is not ' +
        'supported and is rejected because it can be used to exhaust memory during parsing.',
    );
  }
  const parser = new XMLParser({
    ignoreAttributes: false,
    attributeNamePrefix: '@_',
    parseAttributeValue: false,
    parseTagValue: false,
    trimValues: true,
    allowBooleanAttributes: true,
  });
  return parser.parse(text) as unknown;
}

// --------------------------------------------------------------- JSON

export function parseJsonText(text: string): unknown {
  const byteLength = Buffer.byteLength(text, 'utf8');
  if (byteLength > MAX_JSON_BYTES) {
    throw new Error(`JSON file too large to parse (${byteLength} bytes > ${MAX_JSON_BYTES} bytes)`);
  }
  return JSON.parse(text) as unknown;
}

// ----------------------------------------------------- format detection

export type DetectedFormat = 'csv' | 'xlsx' | 'xml' | 'json' | 'unknown';

export function detectFormat(
  filename: string,
  mimeType: string,
  textHead?: string,
): DetectedFormat {
  const ext = filename.toLowerCase().split('.').pop() ?? '';
  if (ext === 'xlsx' || ext === 'xlsm' || mimeType.includes('spreadsheetml')) return 'xlsx';
  if (ext === 'xml' || mimeType.includes('xml')) return 'xml';
  if (ext === 'json' || mimeType.includes('json')) return 'json';
  if (
    ext === 'csv' ||
    ext === 'tsv' ||
    ext === 'txt' ||
    mimeType.includes('csv') ||
    mimeType.includes('text')
  ) {
    return 'csv';
  }
  const head = (textHead ?? '').trimStart();
  if (head.startsWith('<')) return 'xml';
  if (head.startsWith('{') || head.startsWith('[')) return 'json';
  if (head.length > 0) return 'csv';
  return 'unknown';
}

/**
 * Builds a `ParsedFile` from raw bytes/text, dispatching to the right
 * low-level parser. XLSX requires a Buffer; everything else accepts text.
 */
export async function parseFile(
  filename: string,
  mimeType: string,
  content: Buffer | string,
): Promise<ParsedFile> {
  const isBuffer = Buffer.isBuffer(content);
  if (isBuffer && content.length > MAX_FILE_BYTES) {
    throw new Error(`File too large (${content.length} bytes > ${MAX_FILE_BYTES} bytes)`);
  }
  const format = detectFormat(
    filename,
    mimeType,
    isBuffer ? content.subarray(0, 256).toString('utf8') : content.slice(0, 256),
  );

  if (format === 'xlsx') {
    const buf = isBuffer ? content : Buffer.from(content, 'utf8');
    const rows = await parseXlsx(buf);
    return { filename, mimeType, rows };
  }

  const text = isBuffer ? content.toString('utf8') : content;
  if (Buffer.byteLength(text, 'utf8') > MAX_FILE_BYTES) {
    throw new Error(
      `File too large (${Buffer.byteLength(text, 'utf8')} bytes > ${MAX_FILE_BYTES} bytes)`,
    );
  }

  if (format === 'xml') {
    return { filename, mimeType, text, xml: parseXmlText(stripBom(text)) };
  }
  if (format === 'json') {
    return { filename, mimeType, text, json: parseJsonText(stripBom(text)) };
  }
  // csv / unknown: parse as CSV, tolerant fallback.
  const rows = parseCsv(text);
  return { filename, mimeType, text, rows };
}

// ------------------------------------------------------------ timestamps

export interface TimestampResult {
  date: Date | null;
  /** Set when parsing failed outright. */
  error?: 'MISSING_TIMESTAMP' | 'INVALID_TIMESTAMP' | 'FUTURE_TIMESTAMP';
  message?: string;
}

const ISO_RE =
  /^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})(?::(\d{2})(?:\.(\d+))?)?\s*(Z|[+-]\d{2}:?\d{2})?$/;
const ISO_DATE_ONLY_RE = /^(\d{4})-(\d{2})-(\d{2})$/;
const SLASH_RE =
  /^(\d{1,2})\/(\d{1,2})\/(\d{4})[ T](\d{1,2}):(\d{2})(?::(\d{2}))?\s*(AM|PM|am|pm)?$/;
const DOT_RE = /^(\d{1,2})\.(\d{1,2})\.(\d{4})[ T](\d{1,2}):(\d{2})(?::(\d{2}))?$/;
/** Dash-separated dates, as used e.g. by LibreView's "Device Timestamp" column (DD-MM-YYYY HH:mm). */
const DASH_RE = /^(\d{1,2})-(\d{1,2})-(\d{4})[ T](\d{1,2}):(\d{2})(?::(\d{2}))?\s*(AM|PM|am|pm)?$/;

/** Epoch: 10 digits ~ seconds (until year 2286), 13 digits ~ millis. */
const EPOCH_SECONDS_MIN = 1_000_000_000; // 2001-09-09
const EPOCH_SECONDS_MAX = 4_000_000_000; // 2096-10-02
const EPOCH_MILLIS_MIN = EPOCH_SECONDS_MIN * 1000;
const EPOCH_MILLIS_MAX = EPOCH_SECONDS_MAX * 1000;

/** Excel's epoch is 1899-12-30 (accounting for the classic 1900 leap-year bug). */
const EXCEL_EPOCH_MS = Date.UTC(1899, 11, 30);

function excelSerialToDate(serial: number): Date {
  return new Date(EXCEL_EPOCH_MS + Math.round(serial * 86400 * 1000));
}

/**
 * Computes the UTC offset (in minutes, east-positive) of an IANA timezone at
 * a given instant, using only Intl — no timezone database dependency.
 */
export function tzOffsetMinutes(date: Date, timeZone: string): number {
  if (timeZone === 'UTC' || timeZone === 'Etc/UTC') return 0;
  try {
    const dtf = new Intl.DateTimeFormat('en-US', {
      timeZone,
      hourCycle: 'h23',
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
    const parts = dtf.formatToParts(date);
    const get = (type: string): number => Number(parts.find((p) => p.type === type)?.value ?? '0');
    const asUtc = Date.UTC(
      get('year'),
      get('month') - 1,
      get('day'),
      get('hour') === 24 ? 0 : get('hour'),
      get('minute'),
      get('second'),
    );
    return Math.round((asUtc - date.getTime()) / 60000);
  } catch {
    return 0;
  }
}

/** Interprets a "local" y/m/d/h/mi/s tuple in the given timezone and returns the corresponding UTC instant. */
function localToUtc(
  y: number,
  mo: number,
  d: number,
  h: number,
  mi: number,
  s: number,
  timezone: string,
): Date {
  if (timezone === 'UTC') return new Date(Date.UTC(y, mo - 1, d, h, mi, s));
  // First guess treating the tuple as UTC, then correct using the zone's offset at that instant.
  const guess = new Date(Date.UTC(y, mo - 1, d, h, mi, s));
  const offset = tzOffsetMinutes(guess, timezone);
  return new Date(guess.getTime() - offset * 60000);
}

/**
 * Scans candidate date strings (in `D/M/Y ...` slash form) across a whole
 * file to decide day-first vs month-first once, consistently. A row with a
 * first component > 12 proves day-first; a row with a second component > 12
 * proves month-first. Falls back to the explicit option, then to MDY with a
 * warning when the file is genuinely ambiguous.
 */
export function resolveDateOrder(
  rawValues: readonly string[],
  explicit?: DateOrder,
): { order: DateOrder; assumed: boolean } {
  for (const raw of rawValues) {
    const m = SLASH_RE.exec(raw.trim()) ?? DASH_RE.exec(raw.trim());
    if (!m) continue;
    const a = Number(m[1]);
    const b = Number(m[2]);
    if (a > 12 && b <= 12) return { order: 'DMY', assumed: false };
    if (b > 12 && a <= 12) return { order: 'MDY', assumed: false };
  }
  if (explicit) return { order: explicit, assumed: false };
  return { order: 'MDY', assumed: true };
}

/**
 * Parses a single timestamp value (string or number) supporting ISO 8601,
 * `YYYY-MM-DD HH:mm[:ss]`, slash-separated dates (day/month order resolved
 * via `dateOrder`), Excel serial dates, and epoch seconds/millis.
 *
 * Timestamps more than 24h in the future (relative to `now`) are rejected as
 * FUTURE_TIMESTAMP, since that almost always means a device clock error.
 */
export function parseTimestamp(
  value: string | number | null | undefined,
  opts: { timezone?: string; dateOrder?: DateOrder; now?: Date } = {},
): TimestampResult {
  const timezone = opts.timezone ?? 'UTC';
  const now = opts.now ?? new Date();

  if (value === null || value === undefined || (typeof value === 'string' && value.trim() === '')) {
    return { date: null, error: 'MISSING_TIMESTAMP', message: 'Timestamp is missing.' };
  }

  let date: Date | null = null;

  if (typeof value === 'number') {
    date = parseNumericTimestamp(value, timezone);
  } else {
    const raw = value.trim();
    date = parseStringTimestamp(raw, timezone, opts.dateOrder);
    if (!date && /^-?\d+(\.\d+)?$/.test(raw)) {
      date = parseNumericTimestamp(Number(raw), timezone);
    }
  }

  if (!date || Number.isNaN(date.getTime())) {
    return {
      date: null,
      error: 'INVALID_TIMESTAMP',
      message: `Could not parse timestamp "${String(value)}".`,
    };
  }

  const maxFuture = now.getTime() + 24 * 3600 * 1000;
  if (date.getTime() > maxFuture) {
    return {
      date,
      error: 'FUTURE_TIMESTAMP',
      message: `Timestamp ${date.toISOString()} is more than 24 hours in the future.`,
    };
  }

  return { date };
}

function parseNumericTimestamp(value: number, timezone: string): Date | null {
  if (!Number.isFinite(value)) return null;
  if (value >= EPOCH_MILLIS_MIN && value <= EPOCH_MILLIS_MAX) return new Date(value);
  if (value >= EPOCH_SECONDS_MIN && value <= EPOCH_SECONDS_MAX) return new Date(value * 1000);
  // Excel serial dates are small numbers (days since 1899-12-30), typically 1..~73000.
  if (value > 0 && value < 200000) {
    void timezone; // Excel serials are treated as the workbook's naive local time == UTC here.
    return excelSerialToDate(value);
  }
  return null;
}

function parseStringTimestamp(raw: string, timezone: string, dateOrder?: DateOrder): Date | null {
  let m = ISO_RE.exec(raw);
  if (m) {
    const [, y, mo, d, h, mi, s, , offset] = m;
    const Y = Number(y);
    const Mo = Number(mo);
    const D = Number(d);
    const H = Number(h);
    const Mi = Number(mi);
    const S = s ? Number(s) : 0;
    if (offset) {
      if (offset === 'Z') return new Date(Date.UTC(Y, Mo - 1, D, H, Mi, S));
      const sign = offset.startsWith('-') ? -1 : 1;
      const clean = offset.slice(1).replace(':', '');
      const oh = Number(clean.slice(0, 2));
      const om = Number(clean.slice(2, 4));
      const utcMs = Date.UTC(Y, Mo - 1, D, H, Mi, S) - sign * (oh * 60 + om) * 60000;
      return new Date(utcMs);
    }
    return localToUtc(Y, Mo, D, H, Mi, S, timezone);
  }

  m = ISO_DATE_ONLY_RE.exec(raw);
  if (m) {
    const [, y, mo, d] = m;
    return localToUtc(Number(y), Number(mo), Number(d), 0, 0, 0, timezone);
  }

  m = SLASH_RE.exec(raw) ?? DASH_RE.exec(raw);
  if (m) {
    return resolveAmbiguousDayFirst(m, dateOrder, timezone);
  }

  m = DOT_RE.exec(raw);
  if (m) {
    const [, d, mo, y, h, mi, s] = m;
    return localToUtc(
      Number(y),
      Number(mo),
      Number(d),
      Number(h),
      Number(mi),
      s ? Number(s) : 0,
      timezone,
    );
  }

  return null;
}

/**
 * Shared day/month resolution for SLASH_RE/DASH_RE matches: applies the
 * chosen dateOrder, self-corrects an impossible month/day swap, and handles
 * an optional AM/PM suffix.
 */
function resolveAmbiguousDayFirst(
  m: RegExpExecArray,
  dateOrder: DateOrder | undefined,
  timezone: string,
): Date {
  const [, a, b, y, h, mi, s, ampm] = m;
  const order = dateOrder ?? 'MDY';
  const A = Number(a);
  const B = Number(b);
  let month: number;
  let day: number;
  if (order === 'DMY') {
    day = A;
    month = B;
  } else {
    month = A;
    day = B;
  }
  // If the resolved order is impossible for this row (e.g. month > 12), swap.
  if (month > 12 && day <= 12) {
    const tmp = month;
    month = day;
    day = tmp;
  }
  let H = Number(h);
  if (ampm) {
    const isPm = ampm.toLowerCase() === 'pm';
    if (isPm && H < 12) H += 12;
    if (!isPm && H === 12) H = 0;
  }
  return localToUtc(Number(y), month, day, H, Number(mi), s ? Number(s) : 0, timezone);
}
