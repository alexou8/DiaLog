/**
 * Plain input types the analytics engine consumes. These are deliberately
 * decoupled from Prisma's generated models (only the enum *types* are
 * reused) so the whole `lib/analytics` tree stays pure TypeScript with no
 * database dependency and is trivially unit-testable with plain arrays.
 */
import type { GlucoseContext, MealType, Intensity, GlucoseUnit } from '@prisma/client';
import type { TargetRange } from '@/lib/domain/thresholds';

export type { GlucoseContext, MealType, Intensity };

export interface GlucosePoint {
  id: string;
  takenAt: Date;
  /** Canonical mg/dL, as stored. */
  valueMgdl: number;
  context: GlucoseContext;
}

export interface MealPoint {
  id: string;
  takenAt: Date;
  mealType: MealType;
  carbsG: number | null;
  description: string;
}

export interface ExercisePoint {
  id: string;
  takenAt: Date;
  endedAt: Date | null;
  durationMin: number;
  activity: string;
  intensity: Intensity;
}

export interface SleepPoint {
  id: string;
  takenAt: Date;
  endedAt: Date;
  durationMin: number;
  quality: number | null;
}

export interface MedicationPoint {
  id: string;
  takenAt: Date;
  name: string;
}

export interface MoodPoint {
  id: string;
  takenAt: Date;
  mood: number;
  stress: number | null;
}

export interface AnalyticsInput {
  glucose: GlucosePoint[];
  meals: MealPoint[];
  exercise: ExercisePoint[];
  sleep: SleepPoint[];
  medications: MedicationPoint[];
  moods: MoodPoint[];
  /** IANA timezone name, e.g. "America/Toronto". All day/hour bucketing is done in this zone. */
  timezone: string;
  /**
   * Unit used when a finding is written out as a sentence. Analysis always
   * happens in mg/dL; this only affects wording. Defaults to mg/dL.
   */
  displayUnit?: GlucoseUnit;
  targetRange: TargetRange;
  periodStart: Date;
  periodEnd: Date;
}
