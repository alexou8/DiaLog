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

const LEVEL_ORDER: readonly EvidenceLevel[] = ['INSUFFICIENT', 'EARLY', 'EMERGING', 'CONSISTENT'];

function minLevel(a: EvidenceLevel, b: EvidenceLevel): EvidenceLevel {
  return LEVEL_ORDER.indexOf(a) <= LEVEL_ORDER.indexOf(b) ? a : b;
}

/** Two-sided significance level below which a group difference counts as more than noise. */
export const COMPARISON_ALPHA = 0.05;

/** |Cohen's d| below this is conventionally a negligible effect. */
export const MIN_MEANINGFUL_EFFECT_SIZE = 0.2;

export interface ComparisonEvidence {
  /** Size of each group being compared. */
  nA: number;
  nB: number;
  /** Two-sided p-value for the difference, or null when no test could run. */
  pValue: number | null;
  /** Standardised effect size (Cohen's d), or null when undefined. */
  effectSize: number | null;
}

/**
 * Grades a two-group comparison.
 *
 * `gradeEvidence(nA + nB)` used to label a comparison "Consistent pattern"
 * at n=40 even when one group had 2 observations, and regardless of whether
 * the difference was distinguishable from noise — p=0.9 could still read as
 * a consistent pattern, which is exactly the overclaiming this module exists
 * to prevent. Now:
 *   - the sample size graded is `2 × min(nA, nB)`: a comparison is only as
 *     strong as its smaller group, so 38 vs 2 grades like 2 vs 2, not 40;
 *   - the level is capped at EARLY unless the difference is both
 *     statistically distinguishable (p < COMPARISON_ALPHA, i.e. its 95%
 *     confidence interval excludes zero) and not negligible in size
 *     (|d| >= MIN_MEANINGFUL_EFFECT_SIZE).
 */
export function gradeComparisonEvidence(
  comparison: ComparisonEvidence,
  analysis: keyof typeof EVIDENCE_THRESHOLDS | string,
): EvidenceLevel {
  const smaller = Math.min(comparison.nA, comparison.nB);
  const bySize = gradeEvidence(2 * smaller, analysis);
  if (bySize === 'INSUFFICIENT') return bySize;
  const distinguishable = comparison.pValue !== null && comparison.pValue < COMPARISON_ALPHA;
  const meaningful =
    comparison.effectSize !== null && Math.abs(comparison.effectSize) >= MIN_MEANINGFUL_EFFECT_SIZE;
  return distinguishable && meaningful ? bySize : minLevel(bySize, 'EARLY');
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
