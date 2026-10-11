/**
 * Glucose interpretation layer.
 *
 * Clinical thresholds live here, never inside components, so that they can be
 * revised, localised or personalised without touching the UI. Every band pairs
 * colour with an icon and a plain-language label — colour is never the only
 * carrier of meaning.
 *
 * Reference bands below are general adult, non-pregnant orientation values
 * commonly published by diabetes organisations. They are *reference context*,
 * not a diagnosis, and are always shown alongside the user's own target range.
 */
import type { GlucoseContext } from '@prisma/client';

export type BandId = 'very-low' | 'low' | 'in-range' | 'above-range' | 'very-high';

export interface GlucoseBand {
  id: BandId;
  /** Short label, e.g. "In your target range". */
  label: string;
  /** One sentence of plain-language context. */
  description: string;
  /** Non-colour redundant cue. */
  icon: 'alert' | 'down' | 'check' | 'up';
  /** CSS custom-property token name defined in the design system. */
  tone: 'critical' | 'caution' | 'positive' | 'notice';
  /** True when the band warrants safety messaging rather than a plain note. */
  safetyMessage?: string;
}

export interface TargetRange {
  lowMgdl: number;
  highMgdl: number;
}

export const DEFAULT_TARGET_RANGE: TargetRange = { lowMgdl: 70, highMgdl: 180 };

/**
 * Snap a stored target range to whole mg/dL before comparing readings to it.
 * Targets saved from mmol/L input used to be stored unrounded (3.9 mmol/L →
 * 70.27 mg/dL), which made a reading of exactly 70 — displayed as 3.9, equal
 * to the target — classify as below range. Snapping at comparison time fixes
 * rows already stored that way as well as new ones.
 */
export function normalizeTargetRange(range: TargetRange): TargetRange {
  return { lowMgdl: Math.round(range.lowMgdl), highMgdl: Math.round(range.highMgdl) };
}

/**
 * Snap a stored reading to whole mg/dL before comparing it to the snapped
 * target range. Readings entered or imported in mmol/L are stored unrounded
 * (10.0 mmol/L → 180.18 mg/dL), so comparing them raw against a whole-mg/dL
 * target put a reading equal to the target, as displayed, outside the range
 * (10.0 against a 10.0 high → "above range"). Both sides must be compared at
 * one resolution. Adjacent mmol/L display steps are 1.8 mg/dL apart, so
 * rounding can never merge two distinct mmol/L values into the same mg/dL.
 */
export function comparableMgdl(mgdl: number): number {
  return Math.round(mgdl);
}

/**
 * Absolute thresholds that sit outside the user's chosen target range and
 * carry safety messaging regardless of personal targets.
 */
export const SAFETY_THRESHOLDS = {
  veryLowMgdl: 54,
  veryHighMgdl: 250,
} as const;

const BANDS: Record<BandId, GlucoseBand> = {
  'very-low': {
    id: 'very-low',
    label: 'Well below your target range',
    description: 'This reading is well below the range you set.',
    icon: 'alert',
    tone: 'critical',
    safetyMessage:
      'Readings this low can need prompt attention. Follow the plan you agreed with your healthcare provider, and contact them if this happens repeatedly.',
  },
  low: {
    id: 'low',
    label: 'Below your target range',
    description: 'This reading is below the range you set.',
    icon: 'down',
    tone: 'caution',
  },
  'in-range': {
    id: 'in-range',
    label: 'In your target range',
    description: 'This reading is inside the range you set.',
    icon: 'check',
    tone: 'positive',
  },
  'above-range': {
    id: 'above-range',
    label: 'Above your target range',
    description: 'This reading is above the range you set.',
    icon: 'up',
    tone: 'notice',
  },
  'very-high': {
    id: 'very-high',
    label: 'Well above your target range',
    description: 'This reading is well above the range you set.',
    icon: 'alert',
    tone: 'critical',
    safetyMessage:
      'Readings this high, especially if they repeat or come with symptoms such as thirst, nausea or confusion, are worth raising with your healthcare provider.',
  },
};

/** Classify a canonical mg/dL reading against a user's target range. */
export function classifyGlucose(
  mgdl: number,
  targetRange: TargetRange = DEFAULT_TARGET_RANGE,
): GlucoseBand {
  const range = normalizeTargetRange(targetRange);
  const value = comparableMgdl(mgdl);
  if (value < SAFETY_THRESHOLDS.veryLowMgdl) return BANDS['very-low'];
  if (value < range.lowMgdl) return BANDS.low;
  if (value > SAFETY_THRESHOLDS.veryHighMgdl) return BANDS['very-high'];
  if (value > range.highMgdl) return BANDS['above-range'];
  return BANDS['in-range'];
}

export function allBands(): GlucoseBand[] {
  return [
    BANDS['very-low'],
    BANDS.low,
    BANDS['in-range'],
    BANDS['above-range'],
    BANDS['very-high'],
  ];
}

export const GLUCOSE_CONTEXT_LABELS: Record<GlucoseContext, string> = {
  FASTING: 'Fasting',
  BEFORE_MEAL: 'Before a meal',
  AFTER_MEAL: 'After a meal',
  BEDTIME: 'At bedtime',
  RANDOM: 'Any time',
  UNKNOWN: 'Not recorded',
};

/**
 * Minutes after a meal within which a reading is treated as a post-meal
 * response for association analysis.
 */
export const POST_MEAL_WINDOW_MIN = { start: 60, end: 180 } as const;
