/**
 * Low-level file parsing: format detection, RFC4180 CSV, XLSX (via exceljs),
 * XML (via fast-xml-parser), JSON, and timestamp parsing shared by every
 * connector.
 *
 * This module never talks to a database and never assumes a particular
 * connector's column layout — that interpretation lives in
 * `lib/import/connectors/*`.
 */
import { inflateRawSync } from 'node:zlib';
import type ExcelJS from 'exceljs';
import { XMLParser } from 'fast-xml-parser';
import { timeZoneOffsetMinutes, zonedPartsToUtc } from '@/lib/domain/time';
import {
  ImportLimitError,
  MAX_IMPORT_BYTES,
  MAX_XLSX_CELLS,
  MAX_XLSX_ENTRIES,
  MAX_XLSX_UNCOMPRESSED_BYTES,
  fileTooLargeMessage,
} from './limits';
import type { DateOrder, ParsedFile } from './types';

// ------------------------------------------------------------- size guards

/**
 * Hard ceiling on any file this module will attempt to parse in memory.
 * Kept as an alias so existing imports keep working; the value lives in
 * ./limits so the UI, action and service agree on it.
 */
export const MAX_FILE_BYTES = MAX_IMPORT_BYTES;

/**
 * Ceiling specifically for JSON.parse, which is O(n) memory on top of the
 * string. No looser than the upload ceiling — it used to be 50 MB, which the
 * 20 MB upload ceiling made unreachable.
 */
export const MAX_JSON_BYTES = MAX_IMPORT_BYTES;

// ------------------------------------------------------------------- BOM

export function stripBom(text: string): string {
  if (text.length > 0 && text.charCodeAt(0) === 0xfeff) return text.slice(1);
  return text;
}

// ------------------------------------------------------------ delimiter

const CANDIDATE_DELIMITERS = [',', ';', '\t', '|'] as const;

/**
 * Only the head of the file is sampled. Splitting the whole text to read ten
 * lines allocated one string per line of the file (~300k for a 20 MB CSV)
 * just to throw all but ten away.
 */
const DELIMITER_SAMPLE_CHARS = 64 * 1024;

/**
 * Auto-detects the CSV delimiter by counting occurrences of each candidate
 * outside of quoted spans across the first few lines and picking the most
 * consistent, most frequent one. Defaults to comma.
 */
export function detectDelimiter(text: string): string {
  const truncated = text.length > DELIMITER_SAMPLE_CHARS;
  let lines = text.slice(0, DELIMITER_SAMPLE_CHARS).split(/\r\n|\r|\n/);
  // The last line of a truncated sample is cut part-way; its delimiter count
  // would fail the consistency check, so drop it.
  if (truncated && lines.length > 1) lines = lines.slice(0, -1);
  const sampleLines = lines.slice(0, 10).filter((l) => l.length > 0);
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

/**
 * An XLSX file is a zip, and the upload ceiling only bounds its compressed
 * size: a deflated worksheet of repeated cells compresses ~1000:1, so a 1 MB
 * upload could inflate to a gigabyte inside ExcelJS and kill the function.
 * Three budgets close that, all enforced before or while ExcelJS reads it:
 *
 *  1. Every zip entry is actually inflated with zlib's `maxOutputLength`
 *     against one shared byte budget. The sizes declared in the zip
 *     directory are attacker-controlled, so they are not trusted — the
 *     bounded inflate itself is what stops.
 *  2. While each worksheet is inflated, its `<c>` cell elements are counted
 *     against a cell budget, so ExcelJS never builds an object model for
 *     more cells than that.
 *  3. Converting cells to strings is bounded by total string length. One
 *     shared string can be referenced by every cell, so a few KB of XML can
 *     stand for gigabytes of text once connectors join rows.
 *
 * ExcelJS's streaming reader was tried and rejected: it depends on zip entry
 * order (it crashes on workbooks ExcelJS itself writes, which put
 * workbook.xml last) and spills out-of-order sheets to temp files on disk.
 */
export interface XlsxBudget {
  maxUncompressedBytes: number;
  maxCells: number;
  maxEntries: number;
}

const DEFAULT_XLSX_BUDGET: XlsxBudget = {
  maxUncompressedBytes: MAX_XLSX_UNCOMPRESSED_BYTES,
  maxCells: MAX_XLSX_CELLS,
  maxEntries: MAX_XLSX_ENTRIES,
};

function workbookTooLarge(): ImportLimitError {
  return new ImportLimitError(
    'WORKBOOK_TOO_LARGE',
    'That spreadsheet expands to more data than DiaLog can import at once. Please export a shorter date range, or save it as CSV.',
  );
}

const EOCD_SIGNATURE = 0x06054b50;
const CENTRAL_DIRECTORY_SIGNATURE = 0x02014b50;
const LOCAL_HEADER_SIGNATURE = 0x04034b50;
const WORKSHEET_ENTRY = /^xl\/worksheets\/[^/]+\.xml$/;

/** Counts `<c ` / `<c>` / `<c/` (cell elements) in worksheet XML bytes. */
function countCellElements(xml: Buffer): number {
  let count = 0;
  for (let i = xml.indexOf('<c'); i >= 0; i = xml.indexOf('<c', i + 2)) {
    const next = xml[i + 2];
    // space, tab, CR, LF, '>' or '/': excludes <col>, <cols>, <cfRule> etc.
    if (next === 0x20 || next === 0x09 || next === 0x0d || next === 0x0a) count++;
    else if (next === 0x3e || next === 0x2f) count++;
  }
  return count;
}

/**
 * Walks the zip central directory and inflates every entry against a shared
 * byte budget, counting worksheet cells against the cell budget. Throws
 * ImportLimitError when a budget is exceeded, and a plain Error when the
 * file is not a readable zip.
 */
export function assertXlsxWithinBudget(
  buf: Buffer,
  budget: XlsxBudget = DEFAULT_XLSX_BUDGET,
): void {
  // The end-of-central-directory record is 22 bytes plus a comment of up to
  // 64 KB, so search backwards from the end within that window.
  let eocd = -1;
  for (let i = buf.length - 22; i >= Math.max(0, buf.length - 22 - 0xffff); i--) {
    if (buf.readUInt32LE(i) === EOCD_SIGNATURE) {
      eocd = i;
      break;
    }
  }
  if (eocd < 0) throw new Error('Not a zip archive.');

  const entryCount = buf.readUInt16LE(eocd + 10);
  const directoryOffset = buf.readUInt32LE(eocd + 16);
  // ZIP64 markers. No spreadsheet within the upload ceiling needs ZIP64.
  if (entryCount === 0xffff || directoryOffset === 0xffffffff) throw workbookTooLarge();
  if (entryCount > budget.maxEntries) throw workbookTooLarge();

  let remaining = budget.maxUncompressedBytes;
  let cells = 0;
  let p = directoryOffset;
  for (let n = 0; n < entryCount; n++) {
    if (p + 46 > buf.length || buf.readUInt32LE(p) !== CENTRAL_DIRECTORY_SIGNATURE) {
      throw new Error('Corrupt zip central directory.');
    }
    const flags = buf.readUInt16LE(p + 8);
    const method = buf.readUInt16LE(p + 10);
    const compressedSize = buf.readUInt32LE(p + 20);
    const nameLength = buf.readUInt16LE(p + 28);
    const extraLength = buf.readUInt16LE(p + 30);
    const commentLength = buf.readUInt16LE(p + 32);
    const localOffset = buf.readUInt32LE(p + 42);
    const name = buf.toString('utf8', p + 46, p + 46 + nameLength);
    p += 46 + nameLength + extraLength + commentLength;

    if (flags & 0x1) throw new Error('Encrypted zip entries are not supported.');
    if (compressedSize === 0xffffffff || localOffset === 0xffffffff) throw workbookTooLarge();
    if (localOffset + 30 > buf.length || buf.readUInt32LE(localOffset) !== LOCAL_HEADER_SIGNATURE) {
      throw new Error('Corrupt zip local header.');
    }
    const dataStart =
      localOffset + 30 + buf.readUInt16LE(localOffset + 26) + buf.readUInt16LE(localOffset + 28);
    const dataEnd = dataStart + compressedSize;
    if (dataEnd > buf.length) throw new Error('Truncated zip entry.');
    const data = buf.subarray(dataStart, dataEnd);

    let inflated: Buffer;
    if (method === 0) {
      inflated = data;
    } else if (method === 8) {
      try {
        // +1 so that "exactly fills the budget" and "exceeds it" differ.
        inflated = inflateRawSync(data, { maxOutputLength: remaining + 1 });
      } catch (error) {
        if ((error as { code?: string }).code === 'ERR_BUFFER_TOO_LARGE') throw workbookTooLarge();
        throw new Error('Corrupt zip entry.');
      }
    } else {
      throw new Error('Unsupported zip compression method.');
    }
    if (inflated.length > remaining) throw workbookTooLarge();
    remaining -= inflated.length;
    if (WORKSHEET_ENTRY.test(name)) {
      cells += countCellElements(inflated);
      if (cells > budget.maxCells) throw workbookTooLarge();
    }
  }
}

type ExcelJsModule = typeof import('exceljs');

/**
 * ExcelJS is loaded only when a spreadsheet is actually imported, so the
 * CSV, JSON and XML paths (and every cold start of the import route) never
 * pay for loading it.
 */
async function loadExcelJs(): Promise<ExcelJsModule> {
  const mod = (await import('exceljs')) as ExcelJsModule & { default?: ExcelJsModule };
  return mod.default ?? mod;
}

/** Parses an XLSX workbook's first worksheet into a row-major grid of strings. */
export async function parseXlsx(
  buffer: ArrayBuffer | Buffer,
  budgetOverrides: Partial<XlsxBudget> = {},
): Promise<string[][]> {
  const budget: XlsxBudget = { ...DEFAULT_XLSX_BUDGET, ...budgetOverrides };
  const buf = Buffer.isBuffer(buffer) ? buffer : Buffer.from(buffer);
  assertXlsxWithinBudget(buf, budget);

  const ExcelJs = await loadExcelJs();
  const workbook = new ExcelJs.Workbook();
  await workbook.xlsx.load(buf as unknown as ExcelJS.Buffer);
  const sheet = workbook.worksheets[0];
  if (!sheet) return [];
  const rows: string[][] = [];
  let textLength = 0;
  sheet.eachRow({ includeEmpty: false }, (row) => {
    const cells: string[] = [];
    // ExcelJS rows are 1-indexed and sparse; walk to the declared cell count.
    for (let c = 1; c <= row.cellCount; c++) {
      const text = cellToString(row.getCell(c).value);
      textLength += text.length;
      if (textLength > budget.maxUncompressedBytes) throw workbookTooLarge();
      cells.push(text);
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
 * during parsing, so the upload size ceiling does not bound it — the blow-up
 * happens after the size check passes. fast-xml-parser has carried a series
 * of advisories for exactly this (GHSA-jmr7-xgp7-cmfj and its incomplete
 * fixes), so the pinned version is only half the defence.
 *
 * DELIBERATE RELAXATION of the AGENTS.md "Import" invariant: this used to
 * reject any DOCTYPE internal subset. Real Apple Health export.xml files
 * begin with `<!DOCTYPE HealthData [ <!ELEMENT ...> <!ATTLIST ...> ]>` — an
 * internal subset with element and attribute declarations but no entities —
 * so that rule rejected every real Apple Health import. The rule is now: a
 * DOCTYPE is accepted only when its internal subset declares no `<!ENTITY`
 * (checked after stripping comments) and it names no external
 * `SYSTEM`/`PUBLIC` DTD. The accepted DOCTYPE is then removed before
 * fast-xml-parser sees the text, so the parser never reads a DTD at all;
 * element and attribute declarations mean nothing to an import.
 *
 * The DOCTYPE is found with a scanner that tracks quotes, comments and the
 * `[`...`]` subset, not a regex: `/<!DOCTYPE[^>[]*\[/` stopped at the `>` in
 * `SYSTEM "a>b"` and missed the subset that followed, while the parser (which
 * does track quotes) read it.
 *
 * `processEntities` is deliberately left at its default: with the DTD gone it
 * only decodes the five predefined entities, and `&amp;` really does appear in
 * Apple Health `sourceName` attributes. Turning it off would store
 * "Dexcom &amp; Friends" verbatim.
 */
function unsafeXml(message: string): ImportLimitError {
  return new ImportLimitError('UNSAFE_XML', message);
}

function startsWithIgnoreCase(text: string, i: number, token: string): boolean {
  return text.slice(i, i + token.length).toUpperCase() === token;
}

/** Index of a DOCTYPE in the prolog (after BOM, whitespace, PIs, comments), or -1. */
function findPrologDoctype(text: string): number {
  let i = text.charCodeAt(0) === 0xfeff ? 1 : 0;
  for (;;) {
    while (i < text.length && /\s/.test(text[i] as string)) i++;
    if (text.startsWith('<?', i)) {
      const close = text.indexOf('?>', i + 2);
      if (close < 0) return -1;
      i = close + 2;
    } else if (text.startsWith('<!--', i)) {
      const close = text.indexOf('-->', i + 4);
      if (close < 0) return -1;
      i = close + 3;
    } else if (startsWithIgnoreCase(text, i, '<!DOCTYPE')) {
      return i;
    } else {
      return -1;
    }
  }
}

interface DoctypeSpan {
  start: number;
  end: number;
  /** Text between `<!DOCTYPE` and the subset (or the closing `>`). */
  header: string;
  /** The internal subset with comments removed, or null when there is none. */
  subset: string | null;
}

function scanDoctype(text: string, start: number): DoctypeSpan {
  let i = start + '<!DOCTYPE'.length;
  let quote: string | null = null;
  let depth = 0;
  let headerEnd = -1;
  let subsetStart = -1;
  let subsetEnd = -1;
  const subsetComments: [number, number][] = [];
  while (i < text.length) {
    const c = text[i] as string;
    if (quote) {
      if (c === quote) quote = null;
      i++;
      continue;
    }
    if (depth > 0 && text.startsWith('<!--', i)) {
      const close = text.indexOf('-->', i + 4);
      if (close < 0) break;
      subsetComments.push([i, close + 3]);
      i = close + 3;
      continue;
    }
    if (depth > 0 && text.startsWith('<?', i)) {
      const close = text.indexOf('?>', i + 2);
      if (close < 0) break;
      i = close + 2;
      continue;
    }
    if (c === '"' || c === "'") {
      quote = c;
    } else if (c === '[') {
      if (depth === 0) {
        headerEnd = i;
        subsetStart = i + 1;
      }
      depth++;
    } else if (c === ']') {
      depth--;
      if (depth < 0) throw unsafeXml('XML file has a malformed DOCTYPE declaration.');
      if (depth === 0) subsetEnd = i;
    } else if (c === '>' && depth === 0) {
      let subset: string | null = null;
      if (subsetStart >= 0) {
        // Rebuild the subset without its comments, so a commented-out
        // declaration neither hides a real one nor counts as one.
        let cursor = subsetStart;
        subset = '';
        for (const [from, to] of subsetComments) {
          subset += text.slice(cursor, from);
          cursor = to;
        }
        subset += text.slice(cursor, subsetEnd);
      }
      return {
        start,
        end: i + 1,
        header: text.slice(start + '<!DOCTYPE'.length, headerEnd >= 0 ? headerEnd : i),
        subset,
      };
    }
    i++;
  }
  throw unsafeXml('XML file has an unterminated DOCTYPE declaration.');
}

/**
 * Applies the DOCTYPE policy above and returns the text with an accepted
 * DOCTYPE removed. Newlines inside it are kept so parser line numbers still
 * match the file.
 */
export function stripSafeDoctype(text: string): string {
  let out = text;
  const start = findPrologDoctype(text);
  if (start >= 0) {
    const span = scanDoctype(text, start);
    // Only a root element name may follow `<!DOCTYPE`; anything else is an
    // external identifier (SYSTEM "..." or PUBLIC "..." "...").
    if (!/^\s+[^\s"'[\]>]+\s*$/.test(span.header)) {
      throw unsafeXml(
        'XML file references an external DTD in its DOCTYPE, which is not supported.',
      );
    }
    if (span.subset !== null && /<!ENTITY/i.test(span.subset)) {
      throw unsafeXml(
        'XML file declares entities in its DOCTYPE internal subset, which is not supported ' +
          'and is rejected because it can be used to exhaust memory during parsing.',
      );
    }
    const keptNewlines = text.slice(span.start, span.end).replace(/[^\n]/g, '');
    out = text.slice(0, span.start) + keptNewlines + text.slice(span.end);
  }
  // Anything DTD-like left over was outside the prolog, where the scanner
  // looked — refuse it rather than let the parser interpret it.
  if (/<!DOCTYPE/i.test(out) || /<!ENTITY/i.test(out)) {
    throw unsafeXml(
      'XML file has a DOCTYPE or entity declaration DiaLog cannot check (outside the prolog), ' +
        'so it is rejected.',
    );
  }
  return out;
}

export function parseXmlText(text: string): unknown {
  const safe = stripSafeDoctype(text);
  const parser = new XMLParser({
    ignoreAttributes: false,
    attributeNamePrefix: '@_',
    parseAttributeValue: false,
    parseTagValue: false,
    trimValues: true,
    allowBooleanAttributes: true,
  });
  return parser.parse(safe) as unknown;
}

// --------------------------------------------------------------- JSON

export function parseJsonText(text: string): unknown {
  const byteLength = Buffer.byteLength(text, 'utf8');
  if (byteLength > MAX_JSON_BYTES) {
    throw new ImportLimitError(
      'FILE_TOO_LARGE',
      `JSON file too large to parse (${byteLength} bytes > ${MAX_JSON_BYTES} bytes)`,
    );
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
 *
 * Throws ImportLimitError when a size or safety limit refuses the file, and a
 * plain Error when the file is simply unreadable.
 */
export async function parseFile(
  filename: string,
  mimeType: string,
  content: Buffer | string,
): Promise<ParsedFile> {
  const isBuffer = Buffer.isBuffer(content);
  const byteLength = isBuffer ? content.length : Buffer.byteLength(content, 'utf8');
  if (byteLength > MAX_FILE_BYTES) {
    throw new ImportLimitError(
      'FILE_TOO_LARGE',
      `File too large (${byteLength} bytes > ${MAX_FILE_BYTES} bytes). ${fileTooLargeMessage()}`,
    );
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

  if (format === 'xml') {
    return { filename, mimeType, text, xml: parseXmlText(stripBom(text)) };
  }
  if (format === 'json') {
    return { filename, mimeType, text, json: parseJsonText(stripBom(text)) };
  }
  // csv / unknown: parse as CSV, tolerant fallback. The raw text is not kept
  // alongside the rows: no connector reads it once rows exist, and holding
  // both kept every CSV in memory twice.
  const rows = parseCsv(text);
  return { filename, mimeType, rows };
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
 * UTC offset (in minutes, east-positive) of an IANA timezone at a given
 * instant. Delegates to lib/domain/time so import and manual entry share one
 * cached, DST-correct implementation. An unknown zone throws RangeError: the
 * old copy silently returned 0, importing every row of a mis-set profile at
 * UTC. prepareImport validates the zone up front so users get a clear error.
 */
export function tzOffsetMinutes(date: Date, timeZone: string): number {
  return timeZoneOffsetMinutes(date, timeZone);
}

/**
 * Rejects calendar-impossible components instead of letting Date.UTC roll
 * them over — "2026-02-30" used to import silently as 2 March.
 */
function isValidDateTime(y: number, mo: number, d: number, h: number, mi: number, s: number) {
  if (mo < 1 || mo > 12 || d < 1 || h > 23 || mi > 59 || s > 59) return false;
  const daysInMonth = new Date(Date.UTC(y, mo, 0)).getUTCDate();
  return d <= daysInMonth;
}

/**
 * Interprets a "local" y/m/d/h/mi/s tuple in the given timezone and returns
 * the corresponding UTC instant, or null for an impossible date. DST gaps and
 * repeated hours follow the policy in lib/domain/time `zonedWallClockToUtcMs`.
 */
function localToUtc(
  y: number,
  mo: number,
  d: number,
  h: number,
  mi: number,
  s: number,
  timezone: string,
): Date | null {
  if (!isValidDateTime(y, mo, d, h, mi, s)) return null;
  return zonedPartsToUtc(y, mo, d, h, mi, s, timezone);
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
      if (!isValidDateTime(Y, Mo, D, H, Mi, S)) return null;
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
): Date | null {
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
