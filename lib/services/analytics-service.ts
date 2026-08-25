/**
 * Application service binding the data layer to the analytics engine and the
 * AI evidence bundle.
 *
 * This is the seam that guarantees the privacy property the product claims:
 * raw records go into `runAnalytics`, and only graded findings and aggregate
 * numbers come out into the bundle that the assistant is allowed to see.
 */
import type { GlucoseUnit, Profile } from '@prisma/client';
import { runAnalytics, type AnalyticsResult } from '@/lib/analytics/engine';
import { buildInsights, type InsightCard } from '@/lib/analytics/insights';
import type { AnalyticsInput } from '@/lib/analytics/types';
import type { EvidenceBundle } from '@/lib/ai/types';
import { loadAnalyticsWindow } from '@/lib/db/health-records';
import { fromMgdl, unitLabel } from '@/lib/domain/units';
import { daysAgo } from '@/lib/domain/time';

export interface AnalyticsWindow {
  from: Date;
  to: Date;
}

export function defaultWindow(days = 30, now = new Date()): AnalyticsWindow {
  return { from: daysAgo(days, now), to: now };
}

export async function analyzeUser(
  userId: string,
  profile: Profile,
  window: AnalyticsWindow = defaultWindow(),
): Promise<{ result: AnalyticsResult; input: AnalyticsInput }> {
  const data = await loadAnalyticsWindow({ userId, from: window.from, to: window.to });

  const input: AnalyticsInput = {
    glucose: data.glucose.map((g) => ({
      id: g.id,
      takenAt: g.takenAt,
      valueMgdl: g.valueMgdl,
      context: g.context,
    })),
    meals: data.meals.map((m) => ({
      id: m.id,
      takenAt: m.takenAt,
      mealType: m.mealType,
      carbsG: m.carbsG,
      description: m.description,
    })),
    exercise: data.exercise.map((e) => ({
      id: e.id,
      takenAt: e.takenAt,
      endedAt: e.endedAt,
      durationMin: e.durationMin,
      activity: e.activity,
      intensity: e.intensity,
    })),
    sleep: data.sleep.map((s) => ({
      id: s.id,
      takenAt: s.takenAt,
      endedAt: s.endedAt,
      durationMin: s.durationMin,
      quality: s.quality,
    })),
    medications: data.medications.map((m) => ({ id: m.id, takenAt: m.takenAt, name: m.name })),
    moods: data.moods.map((m) => ({
      id: m.id,
      takenAt: m.takenAt,
      mood: m.mood,
      stress: m.stress,
    })),
    timezone: profile.timezone,
    displayUnit: profile.glucoseUnit,
    targetRange: { lowMgdl: profile.targetLowMgdl, highMgdl: profile.targetHighMgdl },
    periodStart: window.from,
    periodEnd: window.to,
  };

  return { result: runAnalytics(input), input };
}

export function insightsFor(result: AnalyticsResult): InsightCard[] {
  return buildInsights(result);
}

/** Round to the precision that makes sense for the display unit. */
function display(mgdl: number | null, unit: GlucoseUnit): number | null {
  if (mgdl == null) return null;
  const value = fromMgdl(mgdl, unit);
  return unit === 'MMOLL' ? Math.round(value * 10) / 10 : Math.round(value);
}

/**
 * Build the evidence bundle handed to the assistant. Contains aggregates and
 * graded findings only — never a list of readings.
 */
export function toEvidenceBundle(
  result: AnalyticsResult,
  profile: Profile,
  window: AnalyticsWindow,
): EvidenceBundle {
  const unit = profile.glucoseUnit;
  const s = result.summary;

  return {
    generatedAt: new Date().toISOString(),
    periodStart: window.from.toISOString(),
    periodEnd: window.to.toISOString(),
    units: unitLabel(unit) as 'mg/dL' | 'mmol/L',
    targetRange: {
      low: display(profile.targetLowMgdl, unit) ?? 0,
      high: display(profile.targetHighMgdl, unit) ?? 0,
    },
    summary: {
      readingCount: s.count,
      average: display(s.averageMgdl, unit),
      median: display(s.medianMgdl, unit),
      standardDeviation: display(s.sdMgdl, unit),
      coefficientOfVariationPercent: s.cv == null ? null : Math.round(s.cv * 1000) / 10,
      percentOfReadingsInTargetRange:
        s.percentInRange == null ? null : Math.round(s.percentInRange),
      percentOfReadingsAboveTargetRange:
        s.percentAboveRange == null ? null : Math.round(s.percentAboveRange),
      percentOfReadingsBelowTargetRange:
        s.percentBelowRange == null ? null : Math.round(s.percentBelowRange),
      daysWithReadings: s.daysWithReadings,
      readingsPerDay: s.readingsPerDay == null ? null : Math.round(s.readingsPerDay * 10) / 10,
      trendClassification: result.trend.classification,
      trendDaysOfData: result.trend.sampleSize,
      trendChangePerWeek:
        result.trend.slopeMgdlPerDay == null
          ? null
          : display(result.trend.slopeMgdlPerDay * 7, unit),
      unusualReadingCount: result.anomalies.length,
    },
    findings: result.findings,
    dataQuality: {
      recordCounts: result.dataQuality.counts,
      coverageDays: result.dataQuality.coverageDays,
      skippedAnalyses: result.dataQuality.skippedAnalyses.map((skipped) => ({
        analysis: skipped.analysis,
        reason: skipped.reason,
      })),
    },
  };
}
