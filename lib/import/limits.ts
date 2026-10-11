/**
 * Import size ceilings, in one place so the parser, the service, the Server
 * Action and the UI copy cannot disagree.
 *
 * Why 20 MB and not the old 100 MB: uploads arrive through a Server Action,
 * whose body limit (`serverActions.bodySizeLimit` in next.config.ts) must be
 * raised to match, and the UI used to promise 100 MB while Next's 1 MB
 * default silently rejected anything bigger. 20 MB holds several years of
 * CGM readings as CSV (~16k rows per MB) while keeping a whole-file in-memory
 * parse comfortably inside a serverless function's memory.
 *
 * This module has no dependencies so client components can import the label.
 */

/** Hard ceiling on an uploaded file, in bytes. */
export const MAX_IMPORT_BYTES = 20 * 1024 * 1024;

/** Human-readable form of MAX_IMPORT_BYTES for user-facing copy. */
export const MAX_IMPORT_LABEL = '20 MB';

/**
 * Budget for what an XLSX may inflate to. The upload ceiling only bounds the
 * compressed bytes; a deflated worksheet of repeated cells compresses
 * ~1000:1, so without this a 1 MB "zip bomb" expands to a gigabyte in memory.
 */
export const MAX_XLSX_UNCOMPRESSED_BYTES = MAX_IMPORT_BYTES * 4;

/** Ceiling on worksheet cells read from one workbook. */
export const MAX_XLSX_CELLS = 500_000;

/** Ceiling on entries in an XLSX zip; real workbooks have a few dozen. */
export const MAX_XLSX_ENTRIES = 1_000;

export type ImportLimitReason = 'FILE_TOO_LARGE' | 'WORKBOOK_TOO_LARGE' | 'UNSAFE_XML';

/**
 * A file was refused because of a size or safety limit, as opposed to being
 * unreadable. Typed so the service can tell the user which limit was hit
 * instead of a generic "could not read that file". The message never
 * includes file contents.
 */
export class ImportLimitError extends Error {
  readonly reason: ImportLimitReason;

  constructor(reason: ImportLimitReason, message: string) {
    super(message);
    this.name = 'ImportLimitError';
    this.reason = reason;
  }
}

/** The user-facing sentence for an over-limit upload. */
export function fileTooLargeMessage(): string {
  return `That file is larger than the ${MAX_IMPORT_LABEL} DiaLog can import at once. Please export a shorter date range, or split the export into smaller files.`;
}
