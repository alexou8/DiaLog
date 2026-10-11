/**
 * Guardrails: everything a model response must pass BEFORE it is displayed.
 *
 * All functions here are pure (no I/O, no provider calls) so they are fully
 * unit-testable. `lib/ai/pipeline.ts` is the only caller in production code.
 */
import { AssistantAnswerSchema, type AssistantAnswer, type ConfidenceLevel } from './schemas';
import type { EvidenceBundle, Finding } from './types';

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
// Text normalisation
// ---------------------------------------------------------------------------

/**
 * Fold text into one canonical form before any pattern runs, so the patterns
 * only have to describe plain lower-case ASCII.
 *
 * Each step closes a bypass found in the 2026-10 review, where 24 of 27
 * constructed dosing/diagnosis outputs got past the filter:
 *  - NFKC: fullwidth "４ units" reads as "4 units".
 *  - Strip \p{Cf}: a zero-width space inside "in​sulin" hid the word.
 *  - Map every \p{Nd} digit to ASCII: Arabic-Indic "٤ units" has no NFKC
 *    mapping, so it needs its own fold.
 *  - Drop combining marks after NFD: "unités" and "unites" match one pattern.
 *  - Curly apostrophes to "'": "you’re diabetic" vs "you're diabetic".
 */
export function normalizeForSafety(text: string): string {
  return text
    .normalize('NFKC')
    .replace(/\p{Cf}/gu, '')
    .replace(/\p{Nd}/gu, (ch) => String(unicodeDigitValue(ch)))
    .normalize('NFD')
    .replace(/\p{M}/gu, '')
    .replace(/[‘’ʼ`´]/g, "'")
    .replace(/[‐-―−]/g, '-')
    .replace(/\s+/g, ' ')
    .toLowerCase()
    .trim();
}

/**
 * Numeric value of a Unicode decimal digit. Every \p{Nd} block is a run of
 * consecutive code points starting at its zero (a few blocks sit back to
 * back, e.g. the mathematical digits), so the value is the distance from the
 * start of the run, mod 10.
 */
function unicodeDigitValue(ch: string): number {
  const cp = ch.codePointAt(0) ?? 48;
  let start = cp;
  while (start > cp - 60 && /\p{Nd}/u.test(String.fromCodePoint(start - 1))) start -= 1;
  return (cp - start) % 10;
}

// ---------------------------------------------------------------------------
// Medical-safety filter
// ---------------------------------------------------------------------------

/**
 * A quantity: digits, a fraction, or a spelled-out number. The model is
 * steered by the user's own question ("write numbers as words", "answer in
 * French"), so "four units" must be caught as surely as "4 units".
 */
const NUMBER_WORD = String.raw`(?:\d+(?:[.,]\d+)?(?:\s*/\s*\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|dozen|half|a few|a couple(?: of)?|several|un|une|deux|trois|quatre|cinq|sept|huit|neuf|dix|uno|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez)`;
const NUM = String.raw`${NUMBER_WORD}(?:[\s-]+(?:and\s+a\s+)?${NUMBER_WORD})?`;

/** Dose units. `mg` excludes mg/dL and `u` must stand alone, so "142 mg/dL" and "you" never match. */
const DOSE_UNIT = String.raw`(?:u|iu|units?|unites?|unidades?|einheiten|mg(?!\s*/\s*d?l)|mcg|μg|ug|micrograms?|milligrams?)(?![a-z])`;

/** Insulin and diabetes medications, generic and brand (lower-case, diacritics stripped). */
const MEDICATION_TERM = String.raw`(?:insulin\w*|insuline|insulina|lantus|humalog|novolog|novorapid|levemir|tresiba|toujeo|fiasp|apidra|basaglar|semglee|admelog|lyumjev|humulin|novolin|ozempic|wegovy|rybelsus|mounjaro|zepbound|trulicity|victoza|byetta|jardiance|farxiga|invokana|januvia|metformin|glucophage|glipizide|glyburide|glimepiride|gliclazide|sulfonylureas?|semaglutide|tirzepatide|dulaglutide|liraglutide|empagliflozin|dapagliflozin|sitagliptin|pioglitazone|glucagon|baqsimi|gvoke)`;
const DOSING_NOUN = String.raw`(?:${MEDICATION_TERM}|doses?|dosage|dosing|units?|unites?|unidades?|medications?|medicaments?|meds?|basal|bolus)`;

/** Action verbs (as stems) that turn a medication mention into an instruction, English plus a small French/Spanish set. */
const DOSING_VERB = String.raw`(?:increas|decreas|adjust|lower|rais|stop|start|chang|reduc|inject|pre-?bolus|bolus|skip|hold|doubl|halv|cut|bump|add|try|tri|switch|correct|titrat|resum|paus|miss|delay|split|boost|tak|took|use|dose|augment|aument|baiss|diminu|prend|pren|toma|tome)\w*`;

/** Conditions a diagnosis would name. */
const CONDITION = String.raw`(?:diabet\w*|pre-?diabet\w*|type\s*(?:1|2|one|two)\b|t[12]d|hypoglyc\w*|hyperglyc\w*|insulin resistan\w*|gastroparesis|neuropath\w*|ketoacidosis|condition|disease|disorder|syndrome)`;

const CLINICIAN = String.raw`(?:doctor|clinician|physician|gp|(?:health\s*care|medical)\s+(?:provider|team|professional)|provider|endocrinologist|endo|nurse|care\s+team|specialist|medical\s+(?:help|attention|advice|care))`;

/** Excludes a clock time ("1:30 am", "1:30:00") from the carb-ratio pattern. */
const NOT_A_TIME = String.raw`(?!\s*(?:[ap]\.?m\b|h\b|o'?clock))(?!:\d)`;

function rx(source: string): RegExp {
  return new RegExp(source, 'iu');
}

/**
 * Regex families for prescriptive/dosing/diagnostic language the assistant
 * must never emit, run over `normalizeForSafety` output. Each entry is
 * deliberately over-broad: a false rejection just falls back to a safe
 * templated answer, while a false negative could ship a dosing instruction
 * to a patient. Do not narrow one without weighing that (AGENTS.md, AI).
 */
const UNSAFE_PATTERNS: { label: string; pattern: RegExp }[] = [
  {
    // "Increase your insulin", "Double your Lantus", "Skip your evening
    // insulin", "Augmentez votre insuline": a verb within a short clause of a
    // dosing noun.
    label: 'dosing-instruction',
    pattern: rx(String.raw`\b${DOSING_VERB}\b[^.?!\n]{0,40}\b${DOSING_NOUN}\b`),
  },
  {
    // "Your Lantus dose should be lowered": the noun first, then the verb.
    label: 'dosing-instruction',
    pattern: rx(
      String.raw`\b${MEDICATION_TERM}\b[^.?!\n]{0,30}\b(?:should|could|can|may|might|needs? to|ought to)\s+(?:be\s+)?${DOSING_VERB}`,
    ),
  },
  {
    // Any quantity next to a dose unit, whatever verb (or none) precedes it:
    // "Take 4 U", "a dose of 10 units", "2 extra units", "quatre unités".
    label: 'dose-quantity',
    pattern: rx(
      String.raw`\b${NUM}\s*-?\s*(?:(?:more|extra|additional|fewer|less)\s+)?${DOSE_UNIT}`,
    ),
  },
  {
    // French/Spanish "take X": "prenez deux", "tome 4".
    label: 'dose-quantity',
    pattern: rx(
      String.raw`\b(?:prend\w*|prenez|tom(?:a|e|ar|en))\s+(?:de\s+|la\s+|le\s+)?${NUM}\b`,
    ),
  },
  {
    // "Lantus 20", "20 of your Lantus", "metformin 500".
    label: 'medication-quantity',
    pattern: rx(
      String.raw`\b${MEDICATION_TERM}\b[^.?!\n]{0,20}\b\d|\b\d+(?:[.,]\d+)?\b[^.?!\n]{0,20}\b${MEDICATION_TERM}\b`,
    ),
  },
  {
    // Insulin-delivery verbs and terms are never part of a description of
    // logged glucose data, so any mention is rejected.
    label: 'insulin-delivery-term',
    pattern: rx(String.raw`\b(?:inject\w*|injecter|injectez|pre-?bolus\w*|bolus\w*|basal)\b`),
  },
  {
    // Carb ratios and correction factors: "insulin-to-carb ratio", "1 unit
    // per 10 g", "1:10" near a ratio word.
    label: 'ratio-or-correction',
    pattern: rx(
      [
        String.raw`\binsulin[\s-]*to[\s-]*carb`,
        String.raw`\bcarb(?:ohydrate)?s?[\s-]*(?:to[\s-]*insulin[\s-]*)?ratio`,
        String.raw`\b(?:icr|isf|cir)\b`,
        String.raw`\bcorrection\s+(?:factor|dose|scale|bolus)`,
        String.raw`\bsensitivity\s+factor`,
        String.raw`\bsliding\s+scale`,
        String.raw`\b(?:units?|u)\b[^.?!\n]{0,15}\b(?:per|for every|for each|every)\s+${NUM}\s*(?:g|grams?|carbs?|mg/dl|mmol)`,
        String.raw`\b(?:insulin|carb\w*|ratio|units?)\b[^.?!\n]{0,40}\b1\s*:\s*\d{1,3}\b${NOT_A_TIME}`,
        String.raw`\b1\s*:\s*\d{1,3}\b${NOT_A_TIME}[^.?!\n]{0,40}\b(?:insulin|carb\w*|ratio|units?)\b`,
      ].join('|'),
    ),
  },
  {
    // Hypoglycemia treatment steps ("Eat 15 g of fast carbs"). Only an
    // imperative counts: "you logged 45 g of carbs" is a data description.
    label: 'treatment-instruction',
    pattern: rx(
      String.raw`(?:^|[.!?;:]\s*|\b(?:should|then|and|please|just|now)\s+)(?:eat|drink|have|take|consume|chew|grab|get|try)\s+(?:about\s+|around\s+|roughly\s+|at least\s+)?${NUM}\s*(?:g|grams?)?\s*(?:of\s+)?(?:(?:fast|quick|rapid|simple)[\s-]*(?:acting\s+)?)?(?:carbs?|carbohydrates?|sugar|glucose|juice|soda|candy|candies|tablets?|tabs?|gels?)|\brule\s+of\s+15\b|\b15[\s-]*15\s+rule\b`,
    ),
  },
  {
    label: 'diagnosis',
    pattern: rx(
      String.raw`\byou(?:\s+are|'re|\s+r|\s+have|'ve|\s+(?:might|may|could|likely|probably|must|seem to|appear to)\s+(?:have|be))\b[^.?!\n]{0,40}\b${CONDITION}`,
    ),
  },
  {
    // "Your readings indicate type 2 diabetes", "consistent with prediabetes".
    label: 'diagnosis',
    pattern: rx(
      String.raw`\b(?:indicat\w*|suggest\w*|consistent\s+with|point(?:s|ing)?\s+to|confirm\w*|signs?\s+of|symptoms?\s+of|means?|typical\s+of|characteristic\s+of)\b[^.?!\n]{0,40}\b(?:diabet\w*|pre-?diabet\w*|t[12]d|insulin resistan\w*|gastroparesis|neuropath\w*|ketoacidosis|disease|disorder|syndrome)`,
    ),
  },
  {
    label: 'diagnose-verb',
    pattern: rx(String.raw`\bdiagnos\w*`),
  },
  {
    // "You don't need a doctor", "There's no need to see a doctor", "skip
    // your appointment".
    label: 'skip-doctor',
    pattern: rx(
      [
        String.raw`\b(?:don't|dont|do not|no|never)\s+need\s+(?:to\s+(?:see|visit|call|contact|consult|talk\s+to|speak\s+(?:to|with)|check\s+with)\s+|for\s+)?(?:a\s+|an\s+|your\s+|the\s+|any\s+)?${CLINICIAN}`,
        String.raw`\bno\s+(?:need|reason)\s+(?:to\s+(?:see|visit|call|contact|consult|talk\s+to|bother)|for)\b[^.?!\n]{0,20}\b${CLINICIAN}`,
        String.raw`\b(?:skip|avoid|cancel|postpone)\w*\s+(?:your\s+|the\s+|a\s+|any\s+)?(?:doctor|appointment|clinic|check-?up)`,
        String.raw`\b(?:instead\s+of|rather\s+than)\s+(?:seeing|calling|visiting|contacting|asking)\s+(?:a\s+|your\s+)?${CLINICIAN}`,
      ].join('|'),
    ),
  },
  {
    label: 'clinically-proven',
    pattern: rx(String.raw`\bclinically[\s-]proven\b`),
  },
  {
    label: 'medical-grade',
    pattern: rx(String.raw`\bmedical[\s-]grade\b`),
  },
];

/**
 * Causal language about glucose. The analytics only ever produce
 * associations, so "Evening walks cause your post-dinner glucose to drop"
 * overclaims even when the association is real (review H4: the rule lived
 * only in the prompt, and that sentence passed every guardrail). Checked per
 * sentence: a causal verb plus a glucose word in the same sentence. A negated
 * claim ("does not show that walking causes lower glucose") and a question (a
 * question for a clinician is the hedged form we want) are allowed. Rejecting
 * rather than rewriting: a regex rewrite of model prose would produce text
 * no one reviewed, while the local fallback answer is template-built.
 */
const CAUSAL_VERB =
  /\b(?:caus(?:e|es|ed|ing)|because\s+of|due\s+to|leads?\s+to|led\s+to|results?\s+in|resulted\s+in|drives?|driving|drove|driven\s+by|triggers?|triggered|makes?\s+your|made\s+your|is\s+why|that's\s+why|responsible\s+for)\b/u;
const GLUCOSE_WORD =
  /\b(?:glucose|sugars?|readings?|bg|levels?|spikes?|lows?|highs?|time\s+in\s+range|a1c|hba1c|mg\/dl|mmol)\b/u;
const NEGATION_BEFORE = /(?:\b(?:not|no|never|cannot|without|nor)\b|n't\b)[^.?!]{0,25}$/u;

function hasCausalClaim(normalized: string): boolean {
  for (const raw of normalized.split(/(?<=[.!?])\s+/u)) {
    const sentence = raw.trim();
    if (sentence.length === 0 || sentence.endsWith('?')) continue;
    const match = CAUSAL_VERB.exec(sentence);
    if (!match || !GLUCOSE_WORD.test(sentence)) continue;
    if (NEGATION_BEFORE.test(sentence.slice(0, match.index))) continue;
    return true;
  }
  return false;
}

export interface SafetyCheckResult {
  safe: boolean;
  /** Which pattern labels matched, if any. Labels only, never the matched text. */
  matchedPatterns: string[];
}

/** Labels of every unsafe-language family `text` trips. */
export function findUnsafeLanguage(text: string): string[] {
  const normalized = normalizeForSafety(text);
  const labels: string[] = [];
  for (const { label, pattern } of UNSAFE_PATTERNS) {
    if (pattern.test(normalized)) labels.push(label);
  }
  if (hasCausalClaim(normalized)) labels.push('causal-claim');
  return labels;
}

export interface TextSafetyOptions {
  /**
   * The user's own input for an extraction task (meal or note parsing). A
   * string that is the whole input, or one whole sentence or clause of it, is
   * the user's words echoed back, not model advice, so it is exempt: "took 4
   * units with lunch" in `unparsed` is what they typed. Anything else the
   * model wrote, including a fragment cut out of a clause, is checked.
   */
  echoSource?: string;
}

/**
 * Sentence and clause boundaries: . ; ! ? , and line breaks. A `.` or `,`
 * between two digits is a decimal separator ("2.8", "2,8"), not a boundary.
 */
const CLAUSE_BOUNDARY = /[;!?\n\r]|(?<!\d)[.,]|[.,](?!\d)/u;

/** Normalised for comparison, with surrounding whitespace and punctuation dropped. */
function echoForm(text: string): string {
  return normalizeForSafety(text).replace(/^[\s.,;:!?]+|[\s.,;:!?]+$/gu, '');
}

/**
 * The strings an extraction may echo without a safety check: the whole input
 * and each of its sentences or clauses.
 *
 * Why whole clauses and not any substring (review L9, 2026-10): a substring
 * exemption let the model return "take 10 units" cut out of the user's
 * "don't take 10 units" and have it pass unchecked — a model-controlled
 * bypass that inverts what the user wrote. Cutting only at clause boundaries
 * keeps the user's own sentence intact or checks it.
 */
function echoSegments(source: string): Set<string> {
  const segments = new Set<string>();
  for (const part of [source, ...source.normalize('NFKC').split(CLAUSE_BOUNDARY)]) {
    const form = echoForm(part);
    if (form.length > 0) segments.add(form);
  }
  return segments;
}

/** Scan a list of model-produced strings for unsafe medical language. */
export function checkTextsSafety(
  texts: string[],
  options: TextSafetyOptions = {},
): SafetyCheckResult {
  const echoes = options.echoSource ? echoSegments(options.echoSource) : null;
  const matched: string[] = [];
  for (const text of texts) {
    if (echoes !== null && echoes.has(echoForm(text))) continue;
    matched.push(...findUnsafeLanguage(text));
  }
  return { safe: matched.length === 0, matchedPatterns: [...new Set(matched)] };
}

/** Every string anywhere inside a validated structured output, in document order. */
export function collectStrings(value: unknown): string[] {
  if (typeof value === 'string') return [value];
  if (Array.isArray(value)) return value.flatMap(collectStrings);
  if (value !== null && typeof value === 'object') {
    return Object.values(value as Record<string, unknown>).flatMap(collectStrings);
  }
  return [];
}

function textFieldsOf(answer: AssistantAnswer): string[] {
  return [answer.shortAnswer, ...answer.detail, ...answer.suggestedQuestionsForClinician];
}

/** Scan an AssistantAnswer's text fields for unsafe medical language. */
export function checkMedicalSafety(answer: AssistantAnswer): SafetyCheckResult {
  return checkTextsSafety(textFieldsOf(answer));
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
  'This is based on limited evidence. Treat it as a first hint, not a conclusion.';

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
// Confidence cap
// ---------------------------------------------------------------------------

const CONFIDENCE_RANK: Record<ConfidenceLevel, number> = {
  insufficient: 0,
  low: 1,
  moderate: 2,
  high: 3,
};

/** Mirrors the local provider's mapping, so every provider is held to the same ceiling. */
export const EVIDENCE_TO_CONFIDENCE: Record<Finding['evidenceLevel'], ConfidenceLevel> = {
  CONSISTENT: 'high',
  EMERGING: 'moderate',
  EARLY: 'low',
  INSUFFICIENT: 'insufficient',
};

/**
 * A live model could answer `confidence: 'high'` while citing one EARLY
 * finding, and the UI shows that as a badge (review H3): the local provider
 * mapped evidence to confidence, LLM output did not, which broke "nothing
 * overclaims from a small sample". So:
 *  - INSUFFICIENT findings cannot be cited as support, and are dropped;
 *  - confidence is clamped to the best evidence grade actually cited;
 *  - with nothing left to cite, the answer gets the same 'low' + caveat
 *    treatment as an ungrounded one.
 * Runs after `enforceGrounding`, so every remaining id is in the bundle.
 */
export function enforceConfidenceCap(
  answer: AssistantAnswer,
  bundle: EvidenceBundle,
): GuardrailResult<AssistantAnswer> {
  const notes: string[] = [];
  const byId = new Map(bundle.findings.map((f) => [f.id, f]));
  const cited = answer.citedFindingIds.filter(
    (id) => byId.get(id)?.evidenceLevel !== 'INSUFFICIENT',
  );
  if (cited.length < answer.citedFindingIds.length) {
    notes.push(`dropped ${answer.citedFindingIds.length - cited.length} INSUFFICIENT citation(s)`);
  }

  let next: AssistantAnswer = { ...answer, citedFindingIds: cited };
  if (next.notEnoughData) {
    if (next.confidence !== 'insufficient') notes.push('capped confidence: notEnoughData');
    return { ok: true, value: { ...next, confidence: 'insufficient' }, notes };
  }

  if (cited.length === 0) {
    if (answer.citedFindingIds.length > 0) {
      notes.push('no citations above INSUFFICIENT remained; downgraded confidence');
      next = { ...next, detail: [...next.detail, LIMITED_EVIDENCE_CAVEAT] };
    }
    const ceiling: ConfidenceLevel = 'low';
    if (CONFIDENCE_RANK[next.confidence] > CONFIDENCE_RANK[ceiling]) next.confidence = ceiling;
    return { ok: true, value: next, notes };
  }

  let ceiling: ConfidenceLevel = 'insufficient';
  for (const id of cited) {
    const level = EVIDENCE_TO_CONFIDENCE[byId.get(id)!.evidenceLevel];
    if (CONFIDENCE_RANK[level] > CONFIDENCE_RANK[ceiling]) ceiling = level;
  }
  if (CONFIDENCE_RANK[next.confidence] > CONFIDENCE_RANK[ceiling]) {
    notes.push(`capped confidence at the cited evidence grade (${ceiling})`);
    next = { ...next, confidence: ceiling };
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
 * Run schema validation, medical-safety filtering, grounding, the confidence
 * cap and claim consistency, in that order, over an arbitrary JSON payload
 * claimed to be an AssistantAnswer. Returns `rejected: true` (with the safe
 * fallback answer) if schema validation or the safety filter fails.
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

  const capped = enforceConfidenceCap(grounded.value, bundle);
  notes.push(...capped.notes);

  const claimChecked = enforceClaimConsistency(capped.value, bundle);
  notes.push(...claimChecked.notes);

  return { answer: claimChecked.value, rejected: false, notes };
}
