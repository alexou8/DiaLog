import { describe, expect, it } from 'vitest';
import { buildImportSummary, groupIssues } from '@/lib/import/summary';
import type { DedupeResult } from '@/lib/import/dedupe';
import type { ParseResult, RowIssue } from '@/lib/import/types';

describe('groupIssues', () => {
  it('groups by code and sorts by descending frequency', () => {
    const issues: RowIssue[] = [
      { rowNumber: 1, code: 'MISSING_VALUE', message: 'a' },
      { rowNumber: 2, code: 'INVALID_TIMESTAMP', message: 'b' },
      { rowNumber: 3, code: 'MISSING_VALUE', message: 'c' },
      { rowNumber: 4, code: 'MISSING_VALUE', message: 'd' },
    ];
    const groups = groupIssues(issues);
    expect(groups[0]?.code).toBe('MISSING_VALUE');
    expect(groups[0]?.count).toBe(3);
    expect(groups[1]?.code).toBe('INVALID_TIMESTAMP');
    expect(groups[1]?.count).toBe(1);
  });

  it('caps the number of examples per code', () => {
    const issues: RowIssue[] = Array.from({ length: 20 }, (_, i) => ({
      rowNumber: i + 1,
      code: 'OUT_OF_RANGE' as const,
      message: 'x',
    }));
    const groups = groupIssues(issues);
    expect(groups[0]?.count).toBe(20);
    expect(groups[0]?.examples.length).toBeLessThanOrEqual(5);
  });

  it('returns an empty array when there are no issues', () => {
    expect(groupIssues([])).toEqual([]);
  });

  it('attaches a plain-language explanation to every group', () => {
    const groups = groupIssues([{ rowNumber: 1, code: 'FUTURE_TIMESTAMP', message: 'x' }]);
    expect(groups[0]?.explanation.length).toBeGreaterThan(0);
  });
});

describe('buildImportSummary', () => {
  it('combines parse and dedupe results into totals', () => {
    const parseResult: ParseResult = {
      records: [],
      issues: [{ rowNumber: 1, code: 'MISSING_VALUE', message: 'x' }],
      rowsTotal: 5,
      detectedUnit: 'MGDL',
      warnings: ['assumed something'],
    };
    const dedupe: DedupeResult = {
      fresh: [
        // @ts-expect-error minimal stub for count purposes
        {},
        // @ts-expect-error minimal stub for count purposes
        {},
      ],
      // @ts-expect-error minimal stub for count purposes
      duplicates: [{}],
    };
    const summary = buildImportSummary(parseResult, dedupe);
    expect(summary.rowsTotal).toBe(5);
    expect(summary.rowsImported).toBe(2);
    expect(summary.rowsDuplicate).toBe(1);
    expect(summary.rowsRejected).toBe(1);
    expect(summary.detectedUnit).toBe('MGDL');
    expect(summary.warnings).toEqual(['assumed something']);
    expect(summary.issueGroups).toHaveLength(1);
  });
});
