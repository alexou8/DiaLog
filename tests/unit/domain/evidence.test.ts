/**
 * Comparison grading must follow the smaller group and the strength of the
 * difference, not the combined sample size — otherwise 38 vs 2 observations,
 * or a p=0.9 difference, could be labelled a "Consistent pattern".
 */
import { describe, expect, it } from 'vitest';
import { gradeComparisonEvidence, gradeEvidence } from '@/lib/domain/evidence';

describe('gradeComparisonEvidence', () => {
  it('grades a balanced, clear difference by its sample size', () => {
    expect(
      gradeComparisonEvidence({ nA: 20, nB: 20, pValue: 0.001, effectSize: 1.2 }, 'comparison'),
    ).toBe('CONSISTENT');
  });

  it('grades a lopsided comparison by its smaller group', () => {
    // 38 + 2 = 40 used to reach CONSISTENT; 2 x min(38, 2) = 4 is below early (8).
    expect(gradeEvidence(40, 'comparison')).toBe('CONSISTENT');
    expect(
      gradeComparisonEvidence({ nA: 38, nB: 2, pValue: 0.001, effectSize: 2 }, 'comparison'),
    ).toBe('INSUFFICIENT');
    expect(
      gradeComparisonEvidence({ nA: 30, nB: 5, pValue: 0.001, effectSize: 2 }, 'comparison'),
    ).toBe('EARLY');
  });

  it('caps a difference indistinguishable from noise at EARLY', () => {
    expect(
      gradeComparisonEvidence({ nA: 25, nB: 25, pValue: 0.9, effectSize: 0.03 }, 'comparison'),
    ).toBe('EARLY');
    expect(
      gradeComparisonEvidence({ nA: 25, nB: 25, pValue: null, effectSize: null }, 'comparison'),
    ).toBe('EARLY');
  });

  it('caps a significant but negligible effect at EARLY', () => {
    expect(
      gradeComparisonEvidence({ nA: 25, nB: 25, pValue: 0.01, effectSize: 0.1 }, 'comparison'),
    ).toBe('EARLY');
  });
});
