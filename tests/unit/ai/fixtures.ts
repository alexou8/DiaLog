/**
 * Shared test fixtures for the AI subsystem test suite.
 */
import type { EvidenceBundle, Finding } from '@/lib/ai/types';

export function makeFinding(overrides: Partial<Finding> = {}): Finding {
  return {
    id: 'finding-1',
    kind: 'post_meal_spike',
    statement: 'Your glucose tends to rise after breakfast on weekdays.',
    sampleSize: 25,
    evidenceLevel: 'EMERGING',
    source: 'STATISTICAL',
    metrics: { avgRiseMgdl: 42, count: 25 },
    basis: '25 breakfasts over the last 60 days',
    periodStart: '2026-06-01',
    periodEnd: '2026-08-01',
    ...overrides,
  };
}

export function makeBundle(overrides: Partial<EvidenceBundle> = {}): EvidenceBundle {
  return {
    generatedAt: '2026-08-24T00:00:00.000Z',
    periodStart: '2026-06-01',
    periodEnd: '2026-08-24',
    units: 'mg/dL',
    targetRange: { low: 70, high: 180 },
    summary: { avgGlucose: 142, readingsCount: 310 },
    findings: [makeFinding()],
    dataQuality: {
      recordCounts: { glucose: 310, meals: 120, exercise: 40 },
      coverageDays: 84,
      skippedAnalyses: [],
    },
    ...overrides,
  };
}

export function makeInsufficientBundle(): EvidenceBundle {
  return makeBundle({
    findings: [makeFinding({ id: 'finding-insuff', evidenceLevel: 'INSUFFICIENT', sampleSize: 2 })],
    dataQuality: {
      recordCounts: { glucose: 4, meals: 1, exercise: 0 },
      coverageDays: 3,
      skippedAnalyses: [{ analysis: 'trend', reason: 'fewer than 10 data points' }],
    },
  });
}
