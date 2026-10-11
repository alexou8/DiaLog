/**
 * Orchestration pipeline: consent gate -> build prompt -> call provider ->
 * guardrails -> return the final answer. This is the only place in `lib/ai`
 * that wires a provider, prompts, and guardrails together.
 *
 * Every entry point takes a required `externalConsent`, applies the consent
 * gate itself (`enforceExternalConsent`), and sends every request through
 * `callProvider`, which refuses an external call without consent. The gate
 * used to live in the server action and covered only the bundle, so the
 * question and quick-log text reached the external provider without consent
 * (review C1/C2, 2026-10). A required parameter means the compiler stops a
 * caller that forgets it.
 */
import {
  AIProviderError,
  enforceExternalConsent,
  type AIProvider,
  type CompletionRequest,
  type CompletionResult,
} from './provider';
import type { EvidenceBundle } from './types';
import {
  buildAnswerQuestionSystemPrompt,
  buildMealParseSystemPrompt,
  buildNoteStructureSystemPrompt,
  buildWeeklyNarrativeSystemPrompt,
  type DetailLevel,
} from './prompts';
import {
  AssistantAnswerJsonSchema,
  MealParseJsonSchema,
  MealParseSchema,
  NoteStructureJsonSchema,
  NoteStructureSchema,
  WeeklyNarrativeJsonSchema,
  WeeklyNarrativeSchema,
  type AssistantAnswer,
  type MealParse,
  type NoteStructure,
  type WeeklyNarrative,
} from './schemas';
import {
  applyAssistantAnswerGuardrails,
  checkTextsSafety,
  collectStrings,
  normalizeForSafety,
  safeFallbackAnswer,
} from './guardrails';
import { detectEmergency, emergencyAnswer } from './emergency';
import { redactForProvider } from './redact';
import { LocalProvider } from './providers/local';

const DEFAULT_MAX_TOKENS = 2048;

/**
 * Defensive runtime check: the AI layer must never receive raw record
 * arrays (readings, meals, etc.) — only aggregated `Finding`s and scalar
 * summary values, per `lib/domain/evidence.ts`. This walks the bundle's
 * own declared fields (not arbitrary unknown properties) looking for any
 * array-of-objects value that is not the `findings` array itself, which
 * would indicate raw records leaking into the bundle.
 */
export function assertNoRawRecords(bundle: EvidenceBundle): void {
  const isPlainObjectArray = (value: unknown): value is Record<string, unknown>[] =>
    Array.isArray(value) &&
    value.every((item) => item !== null && typeof item === 'object' && !Array.isArray(item));

  // `summary` must only ever contain scalars.
  for (const [key, value] of Object.entries(bundle.summary)) {
    if (Array.isArray(value) || (value !== null && typeof value === 'object')) {
      throw new Error(
        `EvidenceBundle.summary.${key} contains a non-scalar value — raw records must never reach the AI layer`,
      );
    }
  }

  // `findings` must contain Finding-shaped objects only (checked structurally,
  // not by size — a large findings array is expected and fine).
  if (!isPlainObjectArray(bundle.findings)) {
    throw new Error('EvidenceBundle.findings must be an array of Finding objects');
  }
  for (const f of bundle.findings) {
    if (
      typeof f['id'] !== 'string' ||
      typeof f['statement'] !== 'string' ||
      typeof f['sampleSize'] !== 'number'
    ) {
      throw new Error(
        'EvidenceBundle.findings contains an entry that is not a valid Finding — possible raw record leak',
      );
    }
    // A Finding must never carry a raw array of readings under `metrics`.
    const metrics = f['metrics'];
    if (metrics !== null && typeof metrics === 'object' && !Array.isArray(metrics)) {
      for (const [mKey, mValue] of Object.entries(metrics as Record<string, unknown>)) {
        if (Array.isArray(mValue) || (mValue !== null && typeof mValue === 'object')) {
          throw new Error(
            `EvidenceBundle.findings[].metrics.${mKey} contains a non-scalar value — raw records must never reach the AI layer`,
          );
        }
      }
    }
  }

  // `dataQuality.skippedAnalyses` is the one other array field — must stay
  // small, flat {analysis, reason} objects, never raw records.
  for (const s of bundle.dataQuality.skippedAnalyses) {
    if (typeof s.analysis !== 'string' || typeof s.reason !== 'string') {
      throw new Error('EvidenceBundle.dataQuality.skippedAnalyses contains a malformed entry');
    }
  }
}

// ---------------------------------------------------------------------------
// Shared provider call
// ---------------------------------------------------------------------------

/**
 * The last line before data leaves the process. Unreachable while every entry
 * point gates first, and kept so that a future code path that forgets the
 * gate fails loudly instead of sending free text out. A plain Error, not an
 * AIProviderError, so the local fallback never swallows it.
 */
async function callProvider(
  provider: AIProvider,
  req: CompletionRequest,
  externalConsent: boolean,
): Promise<CompletionResult> {
  if (provider.isExternal && !externalConsent) {
    throw new Error('Refusing to call an external AI provider without consent');
  }
  return provider.complete(req);
}

interface TaskOutcome<T> {
  value: T;
  providerId: string;
  usedFallback: boolean;
}

/**
 * Try the (already consent-gated) provider, then the local provider, then a
 * fixed last-resort value. `accept` validates and safety-checks a candidate
 * and returns null to reject it. A provider error (network, timeout,
 * malformed JSON, refusal) and a rejected output are treated alike: the user
 * always gets something safe. The local provider is called at most once.
 */
async function runTask<T>(options: {
  provider: AIProvider;
  req: CompletionRequest;
  externalConsent: boolean;
  accept: (json: unknown) => T | null;
  lastResort: () => T;
}): Promise<TaskOutcome<T>> {
  const { provider, req, externalConsent, accept, lastResort } = options;

  const attempt = async (p: AIProvider): Promise<{ value: T; providerId: string } | null> => {
    try {
      const result = await callProvider(p, req, externalConsent);
      const value = accept(result.json);
      return value === null ? null : { value, providerId: result.providerId };
    } catch (err) {
      if (err instanceof AIProviderError) return null;
      throw err;
    }
  };

  const first = await attempt(provider);
  if (first) return { ...first, usedFallback: false };

  const local = new LocalProvider();
  if (provider.id !== local.id) {
    const second = await attempt(local);
    if (second) return { ...second, usedFallback: true };
  }
  return { value: lastResort(), providerId: local.id, usedFallback: true };
}

/** Consent gate plus the redaction that goes with it, for a bundle-carrying task. */
function gateBundleTask(
  provider: AIProvider,
  bundle: EvidenceBundle,
  externalConsent: boolean,
): { provider: AIProvider; bundle: EvidenceBundle } {
  const gated = enforceExternalConsent(provider, externalConsent);
  // Belt and braces: after the gate this is a no-op (an external provider
  // here implies consent), but it keeps `redact.ts` on the path, so loosening
  // the gate later cannot on its own put free-text bundle fields in front of
  // an external provider.
  const { bundle: redacted } = redactForProvider(bundle, gated.isExternal, externalConsent);
  return { provider: gated, bundle: redacted };
}

// ---------------------------------------------------------------------------
// answerQuestion
// ---------------------------------------------------------------------------

export interface AnswerQuestionParams {
  question: string;
  bundle: EvidenceBundle;
  detailLevel: DetailLevel;
  provider: AIProvider;
  /** Whether the user has consented to external AI processing. Required: see the file header. */
  externalConsent: boolean;
}

export interface AnswerQuestionResult {
  answer: AssistantAnswer;
  providerId: string;
  usedFallback: boolean;
  /** True when the question described a possible emergency and got the fixed safety answer. */
  emergency: boolean;
}

export async function answerQuestion(params: AnswerQuestionParams): Promise<AnswerQuestionResult> {
  const { question, detailLevel, externalConsent } = params;

  // Before anything else, including the bundle check: a possible emergency
  // gets the fixed safety answer and never reaches a model (see emergency.ts).
  if (detectEmergency(question).emergency) {
    return { answer: emergencyAnswer(), providerId: 'local', usedFallback: false, emergency: true };
  }

  assertNoRawRecords(params.bundle);
  const { provider, bundle } = gateBundleTask(params.provider, params.bundle, externalConsent);

  const req: CompletionRequest = {
    system: buildAnswerQuestionSystemPrompt(bundle, detailLevel),
    messages: [{ role: 'user', content: question }],
    responseSchema: AssistantAnswerJsonSchema,
    maxTokens: DEFAULT_MAX_TOKENS,
    temperature: 0.2,
  };

  const outcome = await runTask({
    provider,
    req,
    externalConsent,
    accept: (json) => {
      const guardrailed = applyAssistantAnswerGuardrails(json, params.bundle);
      return guardrailed.rejected ? null : guardrailed.answer;
    },
    lastResort: safeFallbackAnswer,
  });
  return {
    answer: outcome.value,
    providerId: outcome.providerId,
    usedFallback: outcome.usedFallback,
    emergency: false,
  };
}

// ---------------------------------------------------------------------------
// parseMealText
// ---------------------------------------------------------------------------

export interface ParseMealTextParams {
  text: string;
  provider: AIProvider;
  /** Whether the user has consented to external AI processing. Required: see the file header. */
  externalConsent: boolean;
}

export interface ParseMealTextResult {
  parse: MealParse;
  providerId: string;
  usedFallback: boolean;
  /**
   * The text described a possible emergency ("BG 2.8 and shaking, had juice").
   * The caller must show `emergencyAnswer` ahead of the proposal.
   */
  emergency: boolean;
  /** The same fixed safety template `answerQuestion` returns; set only when `emergency`. */
  emergencyAnswer?: AssistantAnswer;
}

/**
 * Meal and note parsing output is shown to the user and then saved, so it
 * gets the same medical-safety scan as an answer (review M1: only
 * `answerQuestion` was filtered). A string that is the user's whole text, or
 * one whole sentence or clause of it, is exempt; anything else is checked.
 */
export async function parseMealText(params: ParseMealTextParams): Promise<ParseMealTextResult> {
  const { text, externalConsent } = params;
  // Quick-log text is free text too, and someone logging "BG 2.8 and shaking"
  // needs the same safety answer as someone asking about it. Triage runs
  // before any provider; on a hit the parse runs on the local engine only, so
  // nothing waits on (or is sent to) a model, and whatever was logged (the
  // juice) still comes back to be saved.
  const emergency = detectEmergency(text).emergency;
  const provider = emergency
    ? new LocalProvider()
    : enforceExternalConsent(params.provider, externalConsent);

  const req: CompletionRequest = {
    system: buildMealParseSystemPrompt(),
    messages: [{ role: 'user', content: text }],
    responseSchema: MealParseJsonSchema,
    maxTokens: DEFAULT_MAX_TOKENS,
    temperature: 0.1,
  };

  const outcome = await runTask<MealParse>({
    provider,
    req,
    externalConsent,
    accept: (json) => {
      const validated = MealParseSchema.safeParse(json);
      if (!validated.success) return null;
      const safety = checkTextsSafety(collectStrings(validated.data), { echoSource: text });
      return safety.safe ? validated.data : null;
    },
    lastResort: () => ({ meals: [], exercise: [], unparsed: [text] }),
  });
  return {
    parse: outcome.value,
    providerId: outcome.providerId,
    usedFallback: outcome.usedFallback,
    emergency,
    ...(emergency ? { emergencyAnswer: emergencyAnswer() } : {}),
  };
}

// ---------------------------------------------------------------------------
// summarizePeriod (weekly narrative)
// ---------------------------------------------------------------------------

export interface SummarizePeriodParams {
  bundle: EvidenceBundle;
  detailLevel: DetailLevel;
  provider: AIProvider;
  /** Whether the user has consented to external AI processing. Required: see the file header. */
  externalConsent: boolean;
}

export interface SummarizePeriodResult {
  narrative: WeeklyNarrative;
  providerId: string;
  usedFallback: boolean;
}

function notEnoughDataNarrative(): WeeklyNarrative {
  return {
    headline: 'Not enough data yet for a weekly summary',
    sections: [],
    whatChanged: 'Not enough data yet to say what changed.',
    whatWentWell: 'Not enough data yet to highlight what went well.',
    whatToExploreNext: 'Keep logging consistently.',
    questionsForClinician: [],
  };
}

export async function summarizePeriod(
  params: SummarizePeriodParams,
): Promise<SummarizePeriodResult> {
  const { detailLevel, externalConsent } = params;
  assertNoRawRecords(params.bundle);
  const { provider, bundle } = gateBundleTask(params.provider, params.bundle, externalConsent);

  const req: CompletionRequest = {
    system: buildWeeklyNarrativeSystemPrompt(bundle, detailLevel),
    messages: [{ role: 'user', content: 'Summarize this period.' }],
    responseSchema: WeeklyNarrativeJsonSchema,
    maxTokens: DEFAULT_MAX_TOKENS,
    temperature: 0.3,
  };

  const outcome = await runTask<WeeklyNarrative>({
    provider,
    req,
    externalConsent,
    accept: (json) => {
      const validated = WeeklyNarrativeSchema.safeParse(json);
      if (!validated.success) return null;
      return checkTextsSafety(collectStrings(validated.data)).safe ? validated.data : null;
    },
    lastResort: notEnoughDataNarrative,
  });
  return {
    narrative: outcome.value,
    providerId: outcome.providerId,
    usedFallback: outcome.usedFallback,
  };
}

// ---------------------------------------------------------------------------
// parseNoteText (bonus: NoteStructure task, wired the same way)
// ---------------------------------------------------------------------------

export interface ParseNoteTextParams {
  text: string;
  provider: AIProvider;
  /** Whether the user has consented to external AI processing. Required: see the file header. */
  externalConsent: boolean;
}

export interface ParseNoteTextResult {
  structure: NoteStructure;
  providerId: string;
  usedFallback: boolean;
}

function numbersIn(text: string): Set<number> {
  const out = new Set<number>();
  for (const m of normalizeForSafety(text).matchAll(/\d+(?:[.,]\d+)?/g)) {
    out.add(Number(m[0].replace(',', '.')));
  }
  return out;
}

/**
 * A medication candidate may only carry numbers the user actually wrote. A
 * dose the model inferred ("took my insulin" becoming 10 units) would be
 * saved as if the user had logged it (review M1).
 */
function medicationNumbersAreFromInput(structure: NoteStructure, input: string): boolean {
  const allowed = numbersIn(input);
  for (const candidate of structure.candidates) {
    if (candidate.type !== 'medication') continue;
    const values = [
      candidate.summary,
      ...candidate.fields.map((f) => (f.value === null ? '' : String(f.value))),
    ];
    for (const value of values) {
      for (const n of numbersIn(value)) if (!allowed.has(n)) return false;
    }
  }
  return true;
}

export async function parseNoteText(params: ParseNoteTextParams): Promise<ParseNoteTextResult> {
  const { text, externalConsent } = params;
  // Quick-log text is free text too, and someone logging "BG 2.8 and shaking"
  // needs the same safety answer as someone asking about it. Triage runs
  // before any provider; on a hit the parse runs on the local engine only, so
  // nothing waits on (or is sent to) a model, and whatever was logged (the
  // juice) still comes back to be saved.
  const emergency = detectEmergency(text).emergency;
  const provider = emergency
    ? new LocalProvider()
    : enforceExternalConsent(params.provider, externalConsent);

  const req: CompletionRequest = {
    system: buildNoteStructureSystemPrompt(),
    messages: [{ role: 'user', content: text }],
    responseSchema: NoteStructureJsonSchema,
    maxTokens: DEFAULT_MAX_TOKENS,
    temperature: 0.1,
  };

  const outcome = await runTask<NoteStructure>({
    provider,
    req,
    externalConsent,
    accept: (json) => {
      const validated = NoteStructureSchema.safeParse(json);
      if (!validated.success) return null;
      if (!medicationNumbersAreFromInput(validated.data, text)) return null;
      const safety = checkTextsSafety(collectStrings(validated.data), { echoSource: text });
      return safety.safe ? validated.data : null;
    },
    lastResort: () => ({ candidates: [], unparsed: [text] }),
  });
  return {
    structure: outcome.value,
    providerId: outcome.providerId,
    usedFallback: outcome.usedFallback,
  };
}
