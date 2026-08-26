/**
 * Turns an `AnalyticsResult` into user-facing insight cards matching the
 * shape of the Prisma `Insight` model. Plain language, calm tone: no
 * medical advice, no dose language, no alarming phrasing. Every card
 * carries the evidence it's graded on so the UI can show "why am I seeing
 * this?" from `evidence` alone.
 */
import type { EvidenceLevel, Finding } from '@/lib/domain/evidence';
import { gradeEvidence } from '@/lib/domain/evidence';
import { formatDelta, formatLevel } from './format';
import type { AnalyticsResult } from './engine';

export type InsightSource = 'STATISTICAL' | 'ML' | 'REFERENCE';

export interface InsightCard {
  kind: string;
  title: string;
  summary: string;
  detail: string | null;
  evidenceLevel: EvidenceLevel;
  sampleSize: number;
  source: InsightSource;
  /** Structured bundle backing the "Why am I seeing this?" panel. JSON-serialisable. */
  evidence: Record<string, unknown>;
  periodStart: string;
  periodEnd: string;
}

const FINDING_TITLES: Record<string, string> = {
  'post-meal-carb-bucket': 'Meals and your post-meal glucose',
  'post-dinner-activity': 'Evening activity and your post-dinner glucose',
  'sleep-duration': 'Sleep and your morning glucose',
  'fasting-weekday-weekend': 'Fasting readings: weekdays vs weekends',
  stress: 'Stress and your glucose',
};

function findingToInsight(finding: Finding): InsightCard {
  const detailParts = [finding.basis];
  if (finding.caveats && finding.caveats.length > 0) {
    detailParts.push(`Worth keeping in mind: ${finding.caveats.join(' ')}`);
  }
  return {
    kind: finding.kind,
    title: FINDING_TITLES[finding.kind] ?? 'A pattern in your data',
    summary: finding.statement,
    detail: detailParts.join(' '),
    evidenceLevel: finding.evidenceLevel,
    sampleSize: finding.sampleSize,
    source: finding.source,
    evidence: { metrics: finding.metrics, basis: finding.basis, caveats: finding.caveats ?? [] },
    periodStart: finding.periodStart,
    periodEnd: finding.periodEnd,
  };
}

function periodStrings(result: AnalyticsResult): { periodStart: string; periodEnd: string } {
  // Every finding carries its own period; fall back to "now" bounds if there are none
  // (e.g. a report with anomalies/trend only) so every card still has a valid period.
  const first = result.findings[0];
  if (first) return { periodStart: first.periodStart, periodEnd: first.periodEnd };
  const now = new Date().toISOString();
  return { periodStart: now, periodEnd: now };
}

function trendToInsight(result: AnalyticsResult): InsightCard | null {
  const { trend } = result;
  if (trend.classification === 'not-enough-data') return null;

  const titleByClass: Record<string, string> = {
    improving: 'Your average glucose has been trending down',
    stable: 'Your average glucose has been steady',
    'more-variable': 'Your glucose has been more variable lately',
    rising: 'Your average glucose has been trending up',
  };

  const weeklyChange = trend.slopeMgdlPerDay !== null ? trend.slopeMgdlPerDay * 7 : null;
  const changeText =
    weeklyChange !== null
      ? `about ${formatDelta(weeklyChange, result.displayUnit)} per week ${weeklyChange > 0 ? 'higher' : 'lower'}`
      : 'a change';

  const summary =
    trend.classification === 'stable'
      ? `Your average glucose over this period has stayed fairly steady, without a clear upward or downward trend.`
      : trend.classification === 'more-variable'
        ? `Your day-to-day glucose swings have been trending wider over this period.`
        : `Your average daily glucose has been trending ${trend.classification === 'improving' ? 'down' : 'up'}, by ${changeText} over the period.`;

  return {
    kind: 'trend',
    title: titleByClass[trend.classification] ?? 'Glucose trend',
    summary,
    detail: `Based on ${trend.sampleSize} days with at least one reading, using a robust trend estimate (Theil-Sen slope, Mann-Kendall significance test) rather than a simple straight-line fit.`,
    evidenceLevel: gradeEvidence(trend.sampleSize, 'trend'),
    sampleSize: trend.sampleSize,
    source: 'ML',
    evidence: {
      classification: trend.classification,
      slopeMgdlPerDay: trend.slopeMgdlPerDay,
      mannKendallTau: trend.mannKendallTau,
      mannKendallPValue: trend.mannKendallPValue,
      variabilitySlope: trend.variabilitySlope,
    },
    ...periodStrings(result),
  };
}

function anomaliesToInsight(result: AnalyticsResult): InsightCard | null {
  const { anomalies } = result;
  if (anomalies.length === 0) return null;
  const top = anomalies.slice(0, 5);
  return {
    kind: 'anomaly',
    title: 'A few unusual readings',
    summary: `We noticed ${anomalies.length} reading${anomalies.length === 1 ? '' : 's'} that stood out compared with your own typical readings at that time of day.`,
    detail:
      'These are compared only against your own history at similar times of day, not a population reference range. An unusual reading is not necessarily a problem. It is simply different from your usual pattern.',
    evidenceLevel: gradeEvidence(anomalies.length, 'summary'),
    sampleSize: anomalies.length,
    source: 'ML',
    evidence: {
      flagged: top.map((a) => ({
        takenAt: a.takenAt.toISOString(),
        valueMgdl: a.valueMgdl,
        bucket: a.bucket,
        modifiedZScore: a.modifiedZScore,
        baselineMedianMgdl: a.baselineMedianMgdl,
        baselineSize: a.baselineSize,
      })),
      totalFlagged: anomalies.length,
    },
    ...periodStrings(result),
  };
}

/**
 * A cluster containing one or two days is not a recurring pattern, it is a
 * couple of days. Showing it as "recurring" would overstate what clustering
 * found, so anything below this many days is not surfaced as an insight.
 */
const MIN_DAYS_FOR_RECURRING_PATTERN = 3;

function dayPatternsToInsights(result: AnalyticsResult): InsightCard[] {
  if (!result.dayPatterns) return [];
  return result.dayPatterns
    .filter((cluster) => cluster.size >= MIN_DAYS_FOR_RECURRING_PATTERN)
    .map((cluster) => ({
      kind: 'day-pattern',
      title: `Recurring pattern: ${cluster.label}`,
      summary: `This kind of day (${cluster.label}) showed up ${cluster.size} time${cluster.size === 1 ? '' : 's'} in your logs.`,
      detail: `Typical numbers on these days: average glucose ${formatLevel(cluster.centroid.meanGlucoseMgdl, result.displayUnit)}, ${cluster.centroid.carbsG.toFixed(0)}g carbs logged, ${Math.max(0, cluster.centroid.activityMinutes).toFixed(0)} minutes of activity, ${Math.max(0, cluster.centroid.sleepHours).toFixed(1)}h sleep.`,
      evidenceLevel: gradeEvidence(cluster.size, 'summary'),
      sampleSize: cluster.size,
      source: 'ML',
      evidence: {
        clusterId: cluster.clusterId,
        centroid: cluster.centroid,
        dayCount: cluster.size,
      },
      ...periodStrings(result),
    }));
}

function featureImportanceToInsight(result: AnalyticsResult): InsightCard | null {
  const fi = result.featureImportance;
  if (!fi || fi.coefficients.length === 0) return null;
  const top = fi.coefficients.slice(0, 3);
  const summary = `In your own logs, ${top.map((c) => c.label).join(', ')} move together with your post-meal glucose the most, out of the factors you've logged.`;
  return {
    kind: 'feature-importance',
    title: 'What moves with your post-meal glucose',
    summary,
    detail: `${fi.warning} Based on ${fi.sampleSize} post-meal readings with enough logged context.`,
    evidenceLevel: gradeEvidence(fi.sampleSize, 'model'),
    sampleSize: fi.sampleSize,
    source: 'ML',
    evidence: {
      coefficients: fi.coefficients,
      featuresUsed: fi.featuresUsed,
      featuresDropped: fi.featuresDropped,
      r2: fi.r2,
      ridgeLambda: fi.ridgeLambda,
    },
    ...periodStrings(result),
  };
}

function dataQualityToInsight(result: AnalyticsResult): InsightCard | null {
  const { dataQuality } = result;
  if (dataQuality.skippedAnalyses.length === 0) return null;
  return {
    kind: 'data-quality',
    title: 'Some patterns need more data',
    summary: `We looked, but ${dataQuality.skippedAnalyses.length} analys${dataQuality.skippedAnalyses.length === 1 ? 'is' : 'es'} didn't have enough of your data yet to say anything reliable.`,
    detail: dataQuality.skippedAnalyses.map((s) => `${s.analysis}: ${s.reason}`).join(' '),
    // This card reports what could NOT be concluded. Grading it as a strong
    // pattern would read as the opposite of what it says, so it carries the
    // "not enough data" grade that matches its content.
    evidenceLevel: 'INSUFFICIENT',
    sampleSize: dataQuality.counts.glucose,
    source: 'REFERENCE',
    evidence: {
      skippedAnalyses: dataQuality.skippedAnalyses,
      counts: dataQuality.counts,
      coverageDays: dataQuality.coverageDays,
    },
    ...periodStrings(result),
  };
}

/** Builds every insight card from an `AnalyticsResult`. Order: findings, trend, anomalies, day patterns, feature importance, data quality. */
export function buildInsights(result: AnalyticsResult): InsightCard[] {
  const cards: InsightCard[] = [];
  for (const finding of result.findings) cards.push(findingToInsight(finding));
  const trendCard = trendToInsight(result);
  if (trendCard) cards.push(trendCard);
  const anomalyCard = anomaliesToInsight(result);
  if (anomalyCard) cards.push(anomalyCard);
  cards.push(...dayPatternsToInsights(result));
  const importanceCard = featureImportanceToInsight(result);
  if (importanceCard) cards.push(importanceCard);
  const dataQualityCard = dataQualityToInsight(result);
  if (dataQualityCard) cards.push(dataQualityCard);
  return cards;
}
