/**
 * Orchestrates every analytics sub-module into one `AnalyticsResult`.
 *
 * This is the ONLY shape the AI layer is ever allowed to see — it never
 * gets the raw GlucoseReading / Meal / ... records. Every number here has
 * already been through evidence grading, so nothing downstream can
 * over-claim confidence a small sample doesn't support.
 */
import { EVIDENCE_THRESHOLDS } from '@/lib/domain/evidence';
import type { Finding } from '@/lib/domain/evidence';
import { dayKeyInZone } from '@/lib/domain/time';
import { summarizeGlucose, type GlucoseSummary } from './glucose';
import { runAssociations } from './associations';
import { detectAnomalies, type AnomalyFlag } from './ml/anomaly';
import { clusterDayPatterns, type DayCluster } from './ml/cluster';
import { detectTrend, type TrendResult } from './ml/trend';
import { computeFeatureImportance, type FeatureImportanceResult } from './ml/importance';
import type { AnalyticsInput } from './types';

export interface SkippedAnalysis {
  analysis: string;
  reason: string;
}

export interface DataQualityReport {
  counts: {
    glucose: number;
    meals: number;
    exercise: number;
    sleep: number;
    medications: number;
    moods: number;
  };
  /** Distinct local calendar days with at least one glucose reading. */
  coverageDays: number;
  /** Length of the requested period, in days. */
  periodDays: number;
  averageGapMinutes: number | null;
  maxGapMinutes: number | null;
  skippedAnalyses: SkippedAnalysis[];
}

export interface AnalyticsResult {
  /** Unit the findings were written in. Analysis is always mg/dL. */
  displayUnit: 'MGDL' | 'MMOLL';
  summary: GlucoseSummary;
  findings: Finding[];
  anomalies: AnomalyFlag[];
  trend: TrendResult;
  dayPatterns: DayCluster[] | null;
  featureImportance: FeatureImportanceResult | null;
  dataQuality: DataQualityReport;
}

const ASSOCIATION_KINDS = [
  'post-meal-carb-bucket',
  'post-dinner-activity',
  'sleep-duration',
  'fasting-weekday-weekend',
  'stress',
] as const;

export function runAnalytics(input: AnalyticsInput): AnalyticsResult {
  const summary = summarizeGlucose(
    input.glucose,
    input.targetRange,
    input.timezone,
    input.periodStart,
    input.periodEnd,
  );
  const findings = runAssociations(input);
  const anomalies = detectAnomalies(input.glucose, { timezone: input.timezone });
  const trend = detectTrend(input);
  const dayPatterns = clusterDayPatterns(input);
  const featureImportance = computeFeatureImportance(input);

  const skippedAnalyses: SkippedAnalysis[] = [];

  if (trend.classification === 'not-enough-data') {
    skippedAnalyses.push({
      analysis: 'trend',
      reason: `Fewer than ${EVIDENCE_THRESHOLDS.trend!.early} days with a glucose reading (have ${trend.sampleSize}).`,
    });
  }

  if (dayPatterns === null) {
    skippedAnalyses.push({
      analysis: 'day-patterns',
      reason:
        'Not enough distinct days with a glucose reading to form meaningful day-pattern clusters.',
    });
  }

  if (featureImportance === null) {
    skippedAnalyses.push({
      analysis: 'feature-importance',
      reason: `Fewer than ${EVIDENCE_THRESHOLDS.model!.early} post-meal readings with enough logged context (carbs, activity, sleep, etc.) to fit a personalised model.`,
    });
  }

  const foundKinds = new Set(findings.map((f) => f.kind));
  for (const kind of ASSOCIATION_KINDS) {
    if (!foundKinds.has(kind)) {
      skippedAnalyses.push({
        analysis: `association:${kind}`,
        reason: `Not enough paired data (both groups need at least a few comparable days/meals) to compare for "${kind}".`,
      });
    }
  }

  const coverageDayKeys = new Set(
    input.glucose.map((g) => dayKeyInZone(g.takenAt, input.timezone)),
  );
  const periodDays = Math.max(
    1,
    Math.round((input.periodEnd.getTime() - input.periodStart.getTime()) / 86_400_000),
  );

  const dataQuality: DataQualityReport = {
    counts: {
      glucose: input.glucose.length,
      meals: input.meals.length,
      exercise: input.exercise.length,
      sleep: input.sleep.length,
      medications: input.medications.length,
      moods: input.moods.length,
    },
    coverageDays: coverageDayKeys.size,
    periodDays,
    averageGapMinutes: summary.averageGapMinutes,
    maxGapMinutes: summary.maxGapMinutes,
    skippedAnalyses,
  };

  return {
    displayUnit: input.displayUnit ?? 'MGDL',
    summary,
    findings,
    anomalies,
    trend,
    dayPatterns,
    featureImportance,
    dataQuality,
  };
}
