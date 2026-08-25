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
  range: TargetRange = DEFAULT_TARGET_RANGE,
): GlucoseBand {
  if (mgdl < SAFETY_THRESHOLDS.veryLowMgdl) return BANDS['very-low'];
  if (mgdl < range.lowMgdl) return BANDS.low;
  if (mgdl > SAFETY_THRESHOLDS.veryHighMgdl) return BANDS['very-high'];
  if (mgdl > range.highMgdl) return BANDS['above-range'];
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
