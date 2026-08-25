/**
 * Evidence grading.
 *
 * Every analytical statement DiaLog makes is graded by how much of the user's
 * own data supports it. Thresholds are declared once, here, so they can be
 * reviewed as a set rather than being invented per feature.
 */
import type { EvidenceLevel } from '@prisma/client';

export type { EvidenceLevel };

export interface EvidenceThresholds {
  /** Below this many observations we refuse to draw any conclusion. */
  early: number;
  emerging: number;
  consistent: number;
}

/**
 * Per-analysis minimum sample sizes. These are deliberately conservative:
 * a personal pattern claimed from a handful of readings is not a pattern.
 */
export const EVIDENCE_THRESHOLDS: Record<string, EvidenceThresholds> = {
  /** Simple summaries such as "your average morning reading". */
  summary: { early: 5, emerging: 14, consistent: 30 },
  /** Comparing two groups of days or readings. */
  comparison: { early: 8, emerging: 20, consistent: 40 },
  /** Correlation between a behaviour and a glucose outcome. */
  association: { early: 10, emerging: 24, consistent: 50 },
  /** Trend over time. */
  trend: { early: 10, emerging: 21, consistent: 45 },
  /** Personalised model fitting. */
  model: { early: 30, emerging: 80, consistent: 150 },
};

export function gradeEvidence(
  sampleSize: number,
  analysis: keyof typeof EVIDENCE_THRESHOLDS | string,
): EvidenceLevel {
  const t = EVIDENCE_THRESHOLDS[analysis] ?? EVIDENCE_THRESHOLDS.summary!;
  if (sampleSize >= t.consistent) return 'CONSISTENT';
  if (sampleSize >= t.emerging) return 'EMERGING';
  if (sampleSize >= t.early) return 'EARLY';
  return 'INSUFFICIENT';
}

export const EVIDENCE_LABELS: Record<EvidenceLevel, { label: string; description: string }> = {
  INSUFFICIENT: {
    label: 'Not enough data yet',
    description: 'There are too few records to say anything reliable about this yet.',
  },
  EARLY: {
    label: 'Early signal',
    description:
      'Based on a small number of records. Treat this as a first hint, not a conclusion.',
  },
  EMERGING: {
    label: 'Emerging pattern',
    description: 'Seen often enough to be worth noticing, but still building.',
  },
  CONSISTENT: {
    label: 'Consistent pattern',
    description: 'Seen repeatedly across a substantial number of your records.',
  },
};

/**
 * A structured finding produced by the analytics engine. This is the *only*
 * shape the AI layer is allowed to reason from — it never sees raw records.
 */
export interface Finding {
  id: string;
  kind: string;
  /** Neutral, factual statement of what the data shows. */
  statement: string;
  sampleSize: number;
  evidenceLevel: EvidenceLevel;
  source: 'STATISTICAL' | 'ML' | 'REFERENCE';
  /** Named quantities behind the statement, already unit-converted for display. */
  metrics: Record<string, number | string | null>;
  /** Human-readable description of exactly which records were compared. */
  basis: string;
  /** Period the finding covers. */
  periodStart: string;
  periodEnd: string;
  /** Optional caveats: confounders, data-quality issues, small groups. */
  caveats?: string[];
}
