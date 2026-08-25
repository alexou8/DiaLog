/**
 * Builds the user-facing summary of an import run: totals, per-issue-code
 * breakdown with plain-language explanations, and the assumptions (unit,
 * date order) the parser made along the way.
 */
import type { DedupeResult } from './dedupe';
import type { GlucoseUnit, ParseResult, RowIssue, RowIssueCode } from './types';

export interface IssueGroup {
  code: RowIssueCode;
  count: number;
  explanation: string;
  /** A handful of representative rows, capped so the summary stays readable. */
  examples: RowIssue[];
}

export interface ImportSummary {
  rowsTotal: number;
  rowsImported: number;
  rowsDuplicate: number;
  rowsRejected: number;
  detectedUnit: GlucoseUnit | null;
  warnings: string[];
  issueGroups: IssueGroup[];
}

const ISSUE_EXPLANATIONS: Record<RowIssueCode, string> = {
  MISSING_TIMESTAMP:
    'These rows had no date/time value, so we could not place them on your timeline.',
  INVALID_TIMESTAMP: "These rows had a date/time value we couldn't understand.",
  INVALID_VALUE: "These rows had a value that wasn't a valid number.",
  OUT_OF_RANGE: 'These rows had a value outside what a real reading could plausibly be.',
  UNKNOWN_UNIT: "We couldn't tell whether these values were in mg/dL or mmol/L.",
  MISSING_VALUE: 'These rows were missing a required value.',
  UNSUPPORTED_ROW: "These rows didn't match a kind of record this importer understands.",
  FUTURE_TIMESTAMP:
    'These rows have a date/time more than 24 hours in the future, which usually means a device clock error.',
  PARSE_ERROR: 'These rows could not be parsed at all.',
};

const MAX_EXAMPLES_PER_CODE = 5;

/** Groups issues by code, sorted by descending frequency, each with a bounded example list. */
export function groupIssues(issues: readonly RowIssue[]): IssueGroup[] {
  const byCode = new Map<RowIssueCode, RowIssue[]>();
  for (const issue of issues) {
    const list = byCode.get(issue.code);
    if (list) list.push(issue);
    else byCode.set(issue.code, [issue]);
  }
  return [...byCode.entries()]
    .map(([code, list]) => ({
      code,
      count: list.length,
      explanation: ISSUE_EXPLANATIONS[code],
      examples: list.slice(0, MAX_EXAMPLES_PER_CODE),
    }))
    .sort((a, b) => b.count - a.count);
}

/**
 * Combines a connector's ParseResult with the dedupe pass into the final
 * summary shown to the user (and stored on ImportBatch).
 */
export function buildImportSummary(parseResult: ParseResult, dedupe: DedupeResult): ImportSummary {
  return {
    rowsTotal: parseResult.rowsTotal,
    rowsImported: dedupe.fresh.length,
    rowsDuplicate: dedupe.duplicates.length,
    rowsRejected: parseResult.issues.length,
    detectedUnit: parseResult.detectedUnit,
    warnings: parseResult.warnings,
    issueGroups: groupIssues(parseResult.issues),
  };
}
