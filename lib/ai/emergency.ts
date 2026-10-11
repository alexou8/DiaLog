/**
 * Emergency triage on the user's question, run before any provider call.
 *
 * Why this exists (review H2, 2026-10): "My sugar is 45 and I'm shaking, what
 * do I do?" went down the normal path. The local provider answered with
 * keyword-matched findings or "not enough evidence", and an external model
 * was told to reason only from the bundle. Neither is an answer to someone
 * who may be having a severe low. A glucose app has to expect that question,
 * so it gets a fixed, calm template that points to the person's own care plan
 * and to emergency services, and never reaches a model at all.
 *
 * The template deliberately gives no treatment amounts or dosing: what to do
 * for a low is in the person's care plan, and a number from us is exactly the
 * kind of instruction `guardrails.ts` exists to stop.
 *
 * Like the guardrails, this over-matches on purpose. A false positive costs a
 * user one unhelpful answer; a false negative leaves someone in a severe low
 * reading a chart summary. Pure, no I/O; never logs the question.
 */
import { normalizeForSafety } from './guardrails';
import type { AssistantAnswer } from './schemas';

/** Below these a reading is a severe (level 2) low by the international consensus definition. */
const SEVERE_LOW_MGDL = 54;
const SEVERE_LOW_MMOL = 3.0;
/** Under 20 there is no plausible mg/dL reading in a conscious person, so a bare number that small reads as mmol/L. */
const MIN_PLAUSIBLE_MGDL = 20;

/** Symptom and situation phrases that warrant the emergency answer on their own. */
const EMERGENCY_PHRASES: RegExp[] = [
  /\b(?:passing|passed|pass|blacking|blacked)\s+out\b/u,
  /\b(?:faint(?:ed|ing)?|unconscious|unresponsive|not\s+responding|collapsed?)\b/u,
  /\b(?:seizures?|seizing|convuls\w*|fits?\s+and\s+shaking)\b/u,
  /\b(?:can'?t|cannot|won'?t|couldn'?t)\s+(?:wake|be\s+woken|rouse)\b|\bnot\s+waking\s+up\b/u,
  /\bdka\b|\bketoacidosis\b/u,
  /\bketones?\b[^.?!]{0,20}\b(?:high|large|moderate|very|elevated|positive|over|above)\b/u,
  /\b(?:high|large|moderate|elevated)\s+ketones?\b/u,
  /\bchest\s+pains?\b/u,
  /\bvomit\w*[^.?!]{0,40}\b(?:can'?t|cannot|unable\s+to)\s+keep\b/u,
  /\b(?:can'?t|cannot|unable\s+to)\s+keep\s+(?:anything|food|fluids?|water|liquids?)\s+down\b/u,
  /\boverdos\w*\b|\btoo\s+much\s+insulin\b|\bwrong\s+insulin\b/u,
  /\b(?:suicid\w*|kill\s+myself|end\s+my\s+life|hurt\s+myself|self[\s-]?harm)\b/u,
  /\b(?:meter|cgm|sensor|reading|monitor)\b[^.?!]{0,20}\b(?:says|reads|shows|showing|displays)\s+["']?lo["']?\b/u,
  // A small French set, matching the guardrails' scope.
  /\b(?:inconscient\w*|evanoui\w*|convulsions?|crise\s+d'epilepsie|ne\s+se\s+reveille\s+pas)\b/u,
];

/** Words that make a nearby number a glucose reading rather than a time, dose or carb count. */
const GLUCOSE_CONTEXT =
  /\b(?:glucose|sugar|bg|bs|blood\s+sugar|reading|read|levels?|cgm|sensor|meter|dexcom|libre|number|dropped|dropping|down|fell|falling|crashed|crashing|to|i'?m\s+at|i\s+am\s+at|low|lows|hypo|glycemie|glucemia|azucar)\b[^.?!\d]{0,25}$/u;

/** Units or nouns after a number that mean it is not a glucose reading. */
const NOT_A_READING_AFTER =
  /^\s*(?:[ap]\.?m\b|:|h\b|hrs?\b|hours?\b|mins?\b|minutes?\b|days?\b|weeks?\b|months?\b|years?\b|yrs?\b|g\b|grams?\b|mg\b|mcg\b|ml\b|carbs?\b|%|units?\b|u\b|x\b|times?\b|kg\b|lbs?\b|steps?\b|km\b|miles?\b|cal\w*\b|kcal\b|percent\b|of\b|out\s+of\b|\/)/u;

const NUMBER_WITH_UNIT = /(\d+(?:[.,]\d+)?)\s*(mg\s*\/\s*dl|mmol(?:\s*\/\s*l)?)?/gu;

function hasSevereLowReading(normalized: string): boolean {
  for (const match of normalized.matchAll(NUMBER_WITH_UNIT)) {
    const raw = match[1]!;
    const value = Number(raw.replace(',', '.'));
    if (!Number.isFinite(value)) continue;
    const unit = match[2];
    const index = match.index ?? 0;

    // Part of a larger token ("1:30", "2026-10-10") is never a reading.
    const before = normalized.slice(0, index);
    if (/[\d:.\/-]$/u.test(before)) continue;

    if (unit?.startsWith('mmol')) {
      if (value < SEVERE_LOW_MMOL) return true;
      continue;
    }
    if (unit?.startsWith('mg')) {
      if (value < SEVERE_LOW_MGDL) return true;
      continue;
    }

    // A bare number only counts with glucose wording just before it and
    // nothing after it that makes it a time, dose or amount.
    const after = normalized.slice(index + match[0].length);
    if (NOT_A_READING_AFTER.test(after)) continue;
    if (!GLUCOSE_CONTEXT.test(before)) continue;

    if (value < SEVERE_LOW_MMOL) return true;
    if (value >= MIN_PLAUSIBLE_MGDL && value < SEVERE_LOW_MGDL) return true;
  }
  return false;
}

export interface EmergencyCheckResult {
  emergency: boolean;
}

/** True when the question describes a possible emergency. Never returns or logs the matched text. */
export function detectEmergency(question: string): EmergencyCheckResult {
  const normalized = normalizeForSafety(question);
  if (EMERGENCY_PHRASES.some((pattern) => pattern.test(normalized))) return { emergency: true };
  return { emergency: hasSevereLowReading(normalized) };
}

/** The fixed answer for an emergency question. Contains no amounts and no dosing. */
export function emergencyAnswer(): AssistantAnswer {
  return {
    shortAnswer:
      'What you describe could be urgent. If you or someone with you feels very unwell, call your local emergency number now.',
    detail: [
      'If you think your glucose is low, follow the plan your care team gave you for treating a low, then check again.',
      'If someone is unconscious, having a seizure, or cannot be woken, call emergency services right away.',
      'If you have high ketones, are vomiting, or have chest pain, contact emergency services or your diabetes care team now.',
      'If you are thinking about harming yourself, contact emergency services or a crisis line now.',
      'I am a data assistant and cannot help in an emergency. Once you are safe, I am happy to look at your data with you.',
    ],
    citedFindingIds: [],
    confidence: 'insufficient',
    suggestedQuestionsForClinician: [],
    notEnoughData: true,
  };
}
