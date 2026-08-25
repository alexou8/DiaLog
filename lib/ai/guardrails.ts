/**
 * Guardrails: everything a model response must pass BEFORE it is displayed.
 *
 * All functions here are pure (no I/O, no provider calls) so they are fully
 * unit-testable. `lib/ai/pipeline.ts` is the only caller in production code.
 */
import { AssistantAnswerSchema, type AssistantAnswer, type ConfidenceLevel } from './schemas';
import type { EvidenceBundle } from './types';

export interface GuardrailResult<T> {
  ok: boolean;
  value: T;
  /** Reasons the response was modified or rejected, for logging/telemetry (no health data). */
  notes: string[];
}

// ---------------------------------------------------------------------------
// Schema validation
// ---------------------------------------------------------------------------

/** Validate arbitrary JSON against the AssistantAnswer schema. */
export function validateAssistantAnswer(
  json: unknown,
): { ok: true; value: AssistantAnswer } | { ok: false; error: string } {
  const result = AssistantAnswerSchema.safeParse(json);
  if (result.success) return { ok: true, value: result.data };
  return { ok: false, error: result.error.message };
}

// ---------------------------------------------------------------------------
// Medical-safety filter
// ---------------------------------------------------------------------------

/**
 * Regex families for prescriptive/dosing/diagnostic language the assistant
 * must never emit. Each entry is deliberately conservative (may over-match)
 * because a false rejection just falls back to a safe templated answer,
 * while a false negative could ship unsafe medical advice.
 */
const UNSAFE_PATTERNS: { label: string; pattern: RegExp }[] = [
  {
    label: 'dosing-instruction',
    pattern:
      /\b(increas|decreas|adjust|lower|rais|stop|start|chang|reduc)\w*\b[^.?!\n]{0,40}\b(your\s+)?(insulin|dose|dosage|units?|medication|meds?|metformin|basal|bolus)\b/i,
  },
  {
    label: 'take-units',
    pattern: /\btake\s+\d+(\.\d+)?\s*(units?|mg|mcg|ml|milligrams?)\b/i,
  },
  {
    label: 'diagnosis',
    pattern:
      /\byou\s+(have|are|might have|likely have|probably have)\b[^.?!\n]{0,40}\b(diabetes|diabetic|prediabetes|hypoglycemi\w*|hyperglycemi\w*|insulin resistance|condition|disease|disorder)\b/i,
  },
  {
    label: 'diagnose-verb',
    pattern: /\bdiagnos(e|is|ed|ing)\b/i,
  },
  {
    label: 'skip-doctor',
    pattern: /\byou\s+(don'?t|do not)\s+need\s+to\s+see\s+a\s+doctor\b/i,
  },
  {
    label: 'clinically-proven',
    pattern: /\bclinically[\s-]proven\b/i,
  },
  {
    label: 'medical-grade',
    pattern: /\bmedical[\s-]grade\b/i,
  },
];

export interface SafetyCheckResult {
  safe: boolean;
  /** Which pattern labels matched, if any. */
  matchedPatterns: string[];
}

function textFieldsOf(answer: AssistantAnswer): string[] {
  return [answer.shortAnswer, ...answer.detail, ...answer.suggestedQuestionsForClinician];
}

/** Scan an AssistantAnswer's text fields for unsafe medical language. */
export function checkMedicalSafety(answer: AssistantAnswer): SafetyCheckResult {
  const matched: string[] = [];
  for (const text of textFieldsOf(answer)) {
    for (const { label, pattern } of UNSAFE_PATTERNS) {
      if (pattern.test(text)) matched.push(label);
    }
  }
  return { safe: matched.length === 0, matchedPatterns: [...new Set(matched)] };
}

/** Standard fallback shown when a response is rejected on medical-safety grounds. */
export function safeFallbackAnswer(): AssistantAnswer {
  return {
    shortAnswer:
      'I can share what your data shows, but I cannot give medical advice, dosing guidance, or a diagnosis.',
    detail: [
      'I can share what your data shows, but I cannot give medical advice, dosing guidance, or a diagnosis.',
      'Please bring any questions about medication, dosing, or symptoms to your healthcare provider.',
    ],
    citedFindingIds: [],
    confidence: 'insufficient',
    suggestedQuestionsForClinician: [],
    notEnoughData: true,
  };
}

// ---------------------------------------------------------------------------
// Grounding check
// ---------------------------------------------------------------------------

export const LIMITED_EVIDENCE_CAVEAT =
  'This is based on limited evidence — treat it as a first hint, not a conclusion.';

/**
 * Every `citedFindingIds` entry must exist in the evidence bundle; drop
 * unknown ids. If none remain (and `notEnoughData` was false), downgrade
 * confidence to 'low' and append the standard caveat to `detail`.
 */
export function enforceGrounding(
  answer: AssistantAnswer,
  bundle: EvidenceBundle,
): GuardrailResult<AssistantAnswer> {
  const notes: string[] = [];
  const validIds = new Set(bundle.findings.map((f) => f.id));
  const kept = answer.citedFindingIds.filter((id) => validIds.has(id));
  const dropped = answer.citedFindingIds.length - kept.length;
  if (dropped > 0) notes.push(`dropped ${dropped} unknown citedFindingId(s)`);

  let next: AssistantAnswer = { ...answer, citedFindingIds: kept };

  if (kept.length === 0 && !next.notEnoughData) {
    notes.push('no grounded citations remained; downgraded confidence');
    const downgraded: ConfidenceLevel = 'low';
    next = {
      ...next,
      confidence: downgraded,
      detail: [...next.detail, LIMITED_EVIDENCE_CAVEAT],
    };
  }

  return { ok: true, value: next, notes };
}

// ---------------------------------------------------------------------------
// Claim check
// ---------------------------------------------------------------------------

/** If every finding in the bundle is INSUFFICIENT, force notEnoughData: true regardless of the model's claim. */
export function enforceClaimConsistency(
  answer: AssistantAnswer,
  bundle: EvidenceBundle,
): GuardrailResult<AssistantAnswer> {
  const allInsufficient =
    bundle.findings.length === 0 ||
    bundle.findings.every((f) => f.evidenceLevel === 'INSUFFICIENT');
  if (!allInsufficient || answer.notEnoughData) {
    return { ok: true, value: answer, notes: [] };
  }
  return {
    ok: true,
    value: {
      ...answer,
      notEnoughData: true,
      confidence: 'insufficient',
      citedFindingIds: [],
    },
    notes: ['forced notEnoughData: all findings are INSUFFICIENT'],
  };
}

// ---------------------------------------------------------------------------
// Full pipeline
// ---------------------------------------------------------------------------

export interface ApplyGuardrailsResult {
  answer: AssistantAnswer;
  rejected: boolean;
  notes: string[];
}

/**
 * Run schema validation, medical-safety filtering, grounding, and claim
 * consistency, in that order, over an arbitrary JSON payload claimed to be
 * an AssistantAnswer. Returns `rejected: true` (with the safe fallback
 * answer) if schema validation or the safety filter fails.
 */
export function applyAssistantAnswerGuardrails(
  json: unknown,
  bundle: EvidenceBundle,
): ApplyGuardrailsResult {
  const notes: string[] = [];

  const validated = validateAssistantAnswer(json);
  if (!validated.ok) {
    return {
      answer: safeFallbackAnswer(),
      rejected: true,
      notes: [`schema validation failed: ${validated.error}`],
    };
  }

  const safety = checkMedicalSafety(validated.value);
  if (!safety.safe) {
    return {
      answer: safeFallbackAnswer(),
      rejected: true,
      notes: [`medical safety filter rejected output: ${safety.matchedPatterns.join(', ')}`],
    };
  }

  const grounded = enforceGrounding(validated.value, bundle);
  notes.push(...grounded.notes);

  const claimChecked = enforceClaimConsistency(grounded.value, bundle);
  notes.push(...claimChecked.notes);

  return { answer: claimChecked.value, rejected: false, notes };
}
