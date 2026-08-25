/**
 * Value normalisation shared by connectors: glucose unit inference/conversion
 * and plausibility checks (via lib/domain/units.ts), and free-text context
 * mapping to the GlucoseContext enum.
 */
import { inferGlucoseUnit, isPlausibleGlucose, toMgdl } from '@/lib/domain/units';
import type { GlucoseContext, GlucoseUnit } from '@prisma/client';

export { inferGlucoseUnit };

export interface GlucoseValueResult {
  valueMgdl: number | null;
  /** Set when the value could not be trusted. */
  issue?: 'INVALID_VALUE' | 'OUT_OF_RANGE' | 'UNKNOWN_UNIT';
  message?: string;
}

/**
 * Converts a raw numeric glucose value into canonical mg/dL, given a known or
 * inferred unit, rejecting implausible values (typos, unit mix-ups) rather
 * than silently storing them.
 */
export function normalizeGlucoseValue(
  raw: number | null | undefined,
  unit: GlucoseUnit | null,
): GlucoseValueResult {
  if (raw === null || raw === undefined || !Number.isFinite(raw)) {
    return {
      valueMgdl: null,
      issue: 'INVALID_VALUE',
      message: 'Glucose value is not a valid number.',
    };
  }
  if (!unit) {
    return {
      valueMgdl: null,
      issue: 'UNKNOWN_UNIT',
      message: `Could not determine whether ${raw} is mg/dL or mmol/L.`,
    };
  }
  if (!isPlausibleGlucose(raw, unit)) {
    return {
      valueMgdl: null,
      issue: 'OUT_OF_RANGE',
      message: `Value ${raw} ${unit === 'MMOLL' ? 'mmol/L' : 'mg/dL'} is outside the plausible range for a glucose reading.`,
    };
  }
  return { valueMgdl: toMgdl(raw, unit) };
}

/**
 * Maps free-text context descriptions (as seen in device exports and manual
 * logs) to the GlucoseContext enum. Case/space/punctuation-insensitive.
 */
export function mapGlucoseContext(text: string | null | undefined): GlucoseContext {
  if (!text) return 'UNKNOWN';
  const t = text.toLowerCase().trim();
  if (t.length === 0) return 'UNKNOWN';

  if (/\bfasting\b|\bfast(ed)?\b|before\s*wake|wake\s*up|am\s*fasting/.test(t)) return 'FASTING';
  if (
    /before\s*(breakfast|lunch|dinner|meal|eating)|pre[\s-]?(meal|breakfast|lunch|dinner|prandial)/.test(
      t,
    )
  ) {
    return 'BEFORE_MEAL';
  }
  if (
    /after\s*(breakfast|lunch|dinner|meal|eating)|post[\s-]?(meal|breakfast|lunch|dinner|prandial)/.test(
      t,
    )
  ) {
    return 'AFTER_MEAL';
  }
  if (/bed\s*time|bedtime|before\s*sleep|before\s*bed|night\s*time reading|hs\b/.test(t))
    return 'BEDTIME';
  if (/random|any\s*time|casual|spot\s*check/.test(t)) return 'RANDOM';
  return 'UNKNOWN';
}

/**
 * Convenience wrapper that infers the unit for a whole file (once) when the
 * caller has not been told the unit explicitly, per the heuristic in
 * lib/domain/units.ts: it only trusts values that look unambiguous.
 */
export function detectFileGlucoseUnit(
  values: readonly number[],
  explicit?: GlucoseUnit | null,
): GlucoseUnit | null {
  if (explicit) return explicit;
  return inferGlucoseUnit(values);
}
