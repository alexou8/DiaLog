/**
 * Orchestration pipeline: build prompt -> call provider -> guardrails ->
 * return the final answer. This is the only place in `lib/ai` that wires a
 * provider, prompts, and guardrails together.
 */
import { AIProviderError, type AIProvider, type CompletionRequest } from './provider';
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
import { applyAssistantAnswerGuardrails, safeFallbackAnswer } from './guardrails';
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
// answerQuestion
// ---------------------------------------------------------------------------

export interface AnswerQuestionParams {
  question: string;
  bundle: EvidenceBundle;
  detailLevel: DetailLevel;
  provider: AIProvider;
}

export interface AnswerQuestionResult {
  answer: AssistantAnswer;
  providerId: string;
  usedFallback: boolean;
}

export async function answerQuestion(params: AnswerQuestionParams): Promise<AnswerQuestionResult> {
  const { question, bundle, detailLevel, provider } = params;
  assertNoRawRecords(bundle);

  const req: CompletionRequest = {
    system: buildAnswerQuestionSystemPrompt(bundle, detailLevel),
    messages: [{ role: 'user', content: question }],
    responseSchema: AssistantAnswerJsonSchema,
    maxTokens: DEFAULT_MAX_TOKENS,
    temperature: 0.2,
  };

  return runWithGuardrailsAndFallback(provider, req, bundle);
}

async function runWithGuardrailsAndFallback(
  provider: AIProvider,
  req: CompletionRequest,
  bundle: EvidenceBundle,
): Promise<AnswerQuestionResult> {
  let json: unknown;
  let providerId = provider.id;
  let usedFallback = false;

  try {
    const result = await provider.complete(req);
    json = result.json;
    providerId = result.providerId;
  } catch (err) {
    // Any provider failure (network, timeout, malformed JSON) falls back to
    // the local provider so the user always gets a safe, grounded answer.
    if (!(err instanceof AIProviderError)) throw err;
    const fallback = new LocalProvider();
    const fallbackResult = await fallback.complete(req);
    json = fallbackResult.json;
    providerId = fallbackResult.providerId;
    usedFallback = true;
  }

  const guardrailed = applyAssistantAnswerGuardrails(json, bundle);
  if (guardrailed.rejected && !usedFallback) {
    // Schema/safety rejection on a live model response — retry once against
    // the local provider rather than showing the raw unsafe/invalid output.
    const fallback = new LocalProvider();
    const fallbackResult = await fallback.complete(req);
    const reGuardrailed = applyAssistantAnswerGuardrails(fallbackResult.json, bundle);
    return {
      answer: reGuardrailed.rejected ? safeFallbackAnswer() : reGuardrailed.answer,
      providerId: fallbackResult.providerId,
      usedFallback: true,
    };
  }

  return { answer: guardrailed.answer, providerId, usedFallback };
}

// ---------------------------------------------------------------------------
// parseMealText
// ---------------------------------------------------------------------------

export interface ParseMealTextParams {
  text: string;
  provider: AIProvider;
}

export interface ParseMealTextResult {
  parse: MealParse;
  providerId: string;
  usedFallback: boolean;
}

export async function parseMealText(params: ParseMealTextParams): Promise<ParseMealTextResult> {
  const { text, provider } = params;

  const req: CompletionRequest = {
    system: buildMealParseSystemPrompt(),
    messages: [{ role: 'user', content: text }],
    responseSchema: MealParseJsonSchema,
    maxTokens: DEFAULT_MAX_TOKENS,
    temperature: 0.1,
  };

  let json: unknown;
  let providerId = provider.id;
  let usedFallback = false;

  try {
    const result = await provider.complete(req);
    json = result.json;
    providerId = result.providerId;
  } catch (err) {
    if (!(err instanceof AIProviderError)) throw err;
    const fallback = new LocalProvider();
    const fallbackResult = await fallback.complete(req);
    json = fallbackResult.json;
    providerId = fallbackResult.providerId;
    usedFallback = true;
  }

  const validated = MealParseSchema.safeParse(json);
  if (validated.success) {
    return { parse: validated.data, providerId, usedFallback };
  }

  // Malformed structured output: fall back to a safe "everything unparsed" result.
  const fallback = new LocalProvider();
  const fallbackResult = await fallback.complete(req);
  const reValidated = MealParseSchema.safeParse(fallbackResult.json);
  return {
    parse: reValidated.success ? reValidated.data : { meals: [], exercise: [], unparsed: [text] },
    providerId: fallbackResult.providerId,
    usedFallback: true,
  };
}

// ---------------------------------------------------------------------------
// summarizePeriod (weekly narrative)
// ---------------------------------------------------------------------------

export interface SummarizePeriodParams {
  bundle: EvidenceBundle;
  detailLevel: DetailLevel;
  provider: AIProvider;
}

export interface SummarizePeriodResult {
  narrative: WeeklyNarrative;
  providerId: string;
  usedFallback: boolean;
}

export async function summarizePeriod(
  params: SummarizePeriodParams,
): Promise<SummarizePeriodResult> {
  const { bundle, detailLevel, provider } = params;
  assertNoRawRecords(bundle);

  const req: CompletionRequest = {
    system: buildWeeklyNarrativeSystemPrompt(bundle, detailLevel),
    messages: [{ role: 'user', content: 'Summarize this period.' }],
    responseSchema: WeeklyNarrativeJsonSchema,
    maxTokens: DEFAULT_MAX_TOKENS,
    temperature: 0.3,
  };

  let json: unknown;
  let providerId = provider.id;
  let usedFallback = false;

  try {
    const result = await provider.complete(req);
    json = result.json;
    providerId = result.providerId;
  } catch (err) {
    if (!(err instanceof AIProviderError)) throw err;
    const fallback = new LocalProvider();
    const fallbackResult = await fallback.complete(req);
    json = fallbackResult.json;
    providerId = fallbackResult.providerId;
    usedFallback = true;
  }

  const validated = WeeklyNarrativeSchema.safeParse(json);
  if (validated.success) {
    return { narrative: validated.data, providerId, usedFallback };
  }

  const fallback = new LocalProvider();
  const fallbackResult = await fallback.complete(req);
  const reValidated = WeeklyNarrativeSchema.safeParse(fallbackResult.json);
  return {
    narrative: reValidated.success
      ? reValidated.data
      : {
          headline: 'Not enough data yet for a weekly summary',
          sections: [],
          whatChanged: 'Not enough data yet to say what changed.',
          whatWentWell: 'Not enough data yet to highlight what went well.',
          whatToExploreNext: 'Keep logging consistently.',
          questionsForClinician: [],
        },
    providerId: fallbackResult.providerId,
    usedFallback: true,
  };
}

// ---------------------------------------------------------------------------
// parseNoteText (bonus: NoteStructure task, wired the same way)
// ---------------------------------------------------------------------------

export interface ParseNoteTextParams {
  text: string;
  provider: AIProvider;
}

export interface ParseNoteTextResult {
  structure: NoteStructure;
  providerId: string;
  usedFallback: boolean;
}

export async function parseNoteText(params: ParseNoteTextParams): Promise<ParseNoteTextResult> {
  const { text, provider } = params;

  const req: CompletionRequest = {
    system: buildNoteStructureSystemPrompt(),
    messages: [{ role: 'user', content: text }],
    responseSchema: NoteStructureJsonSchema,
    maxTokens: DEFAULT_MAX_TOKENS,
    temperature: 0.1,
  };

  let json: unknown;
  let providerId = provider.id;
  let usedFallback = false;

  try {
    const result = await provider.complete(req);
    json = result.json;
    providerId = result.providerId;
  } catch (err) {
    if (!(err instanceof AIProviderError)) throw err;
    const fallback = new LocalProvider();
    const fallbackResult = await fallback.complete(req);
    json = fallbackResult.json;
    providerId = fallbackResult.providerId;
    usedFallback = true;
  }

  const validated = NoteStructureSchema.safeParse(json);
  if (validated.success) {
    return { structure: validated.data, providerId, usedFallback };
  }

  const fallback = new LocalProvider();
  const fallbackResult = await fallback.complete(req);
  const reValidated = NoteStructureSchema.safeParse(fallbackResult.json);
  return {
    structure: reValidated.success ? reValidated.data : { candidates: [], unparsed: [text] },
    providerId: fallbackResult.providerId,
    usedFallback: true,
  };
}
