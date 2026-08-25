/**
 * Structural input type for the AI subsystem.
 *
 * The analytics engine (`lib/analytics/engine.ts`, owned by another agent)
 * produces its own `AnalyticsResult` shape. The AI layer never imports it —
 * the app layer is responsible for mapping analytics output onto this
 * `EvidenceBundle` shape before calling into `lib/ai`. This keeps the AI
 * subsystem decoupled from analytics internals and testable in isolation.
 *
 * This is also, per `lib/domain/evidence.ts`, the *only* data shape the AI
 * is allowed to reason from: a bundle of `Finding`s plus a small summary —
 * never raw readings, raw meals, or raw records of any kind.
 */
import type { Finding } from '@/lib/domain/evidence';

export interface EvidenceBundle {
  generatedAt: string;
  periodStart: string;
  periodEnd: string;
  units: 'mg/dL' | 'mmol/L';
  targetRange: { low: number; high: number };
  summary: Record<string, number | string | null>;
  findings: Finding[];
  dataQuality: {
    recordCounts: Record<string, number>;
    coverageDays: number;
    skippedAnalyses: { analysis: string; reason: string }[];
  };
}

export type { Finding };
