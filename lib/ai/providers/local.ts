/**
 * The local provider: a deterministic, NO-NETWORK implementation.
 *
 * This is the DEFAULT provider so the whole product works with zero API
 * keys and zero health data ever leaving the server. It never calls
 * `fetch`; it composes plain-language answers directly from the evidence
 * bundle it is handed, using templates and keyword matching over
 * `Finding.kind` / `Finding.statement`.
 *
 * It produces the same structured JSON shapes the LLM providers produce
 * (see `lib/ai/schemas.ts`), so `lib/ai/pipeline.ts` and the guardrails in
 * `lib/ai/guardrails.ts` can treat every provider identically.
 */
import type { AIProvider, CompletionRequest, CompletionResult } from '../provider';
import type { EvidenceBundle, Finding } from '../types';
import type { ConfidenceLevel } from '../schemas';

const EVIDENCE_RANK: Record<Finding['evidenceLevel'], number> = {
  CONSISTENT: 3,
  EMERGING: 2,
  EARLY: 1,
  INSUFFICIENT: 0,
};

const EVIDENCE_TO_CONFIDENCE: Record<Finding['evidenceLevel'], ConfidenceLevel> = {
  CONSISTENT: 'high',
  EMERGING: 'moderate',
  EARLY: 'low',
  INSUFFICIENT: 'insufficient',
};

/** Very small stopword list, enough to keep keyword overlap meaningful. */
const STOPWORDS = new Set([
  'a',
  'an',
  'the',
  'is',
  'are',
  'was',
  'were',
  'be',
  'been',
  'am',
  'i',
  'my',
  'me',
  'do',
  'does',
  'did',
  'to',
  'of',
  'in',
  'on',
  'at',
  'for',
  'and',
  'or',
  'but',
  'with',
  'about',
  'what',
  'why',
  'how',
  'when',
  'does',
  'this',
  'that',
  'it',
  'you',
  'your',
  'have',
  'has',
  'can',
  'should',
  'would',
  'could',
  'will',
  'so',
  'if',
  'than',
]);

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter((w) => w.length > 1 && !STOPWORDS.has(w));
}

/** Score a finding's relevance to a question by keyword overlap over kind + statement + basis. */
function relevanceScore(question: string, finding: Finding): number {
  const qTokens = new Set(tokenize(question));
  if (qTokens.size === 0) return 0;
  const haystack = tokenize(`${finding.kind} ${finding.statement} ${finding.basis}`);
  let hits = 0;
  for (const token of haystack) {
    if (qTokens.has(token)) hits += 1;
  }
  // Also credit a direct substring match of the finding kind (e.g. "carbs" in "carb_correlation").
  const kindTokens = tokenize(finding.kind);
  for (const kt of kindTokens) {
    if ([...qTokens].some((q) => kt.includes(q) || q.includes(kt))) hits += 1;
  }
  return hits;
}

/**
 * Findings scoring far below the best match are usually only related by a
 * stray shared word ("glucose"), and padding an answer with them makes it look
 * like the question was not understood. Keep the close matches only.
 */
const RELEVANCE_FLOOR_RATIO = 0.5;

function pickRelevantFindings(question: string, findings: Finding[], max = 3): Finding[] {
  const scored = findings
    .map((f) => ({ f, score: relevanceScore(question, f) }))
    .filter(({ score }) => score > 0)
    .sort((a, b) => {
      if (b.score !== a.score) return b.score - a.score;
      return EVIDENCE_RANK[b.f.evidenceLevel] - EVIDENCE_RANK[a.f.evidenceLevel];
    });

  const best = scored[0]?.score ?? 0;
  const floor = Math.max(1, best * RELEVANCE_FLOOR_RATIO);
  return scored
    .filter(({ score }) => score >= floor)
    .slice(0, max)
    .map(({ f }) => f);
}

function overallConfidence(findings: Finding[]): ConfidenceLevel {
  if (findings.length === 0) return 'insufficient';
  let best: Finding['evidenceLevel'] = 'INSUFFICIENT';
  for (const f of findings) {
    if (EVIDENCE_RANK[f.evidenceLevel] > EVIDENCE_RANK[best]) best = f.evidenceLevel;
  }
  return EVIDENCE_TO_CONFIDENCE[best];
}

function buildNotEnoughDataAnswer(bundle: EvidenceBundle): unknown {
  const suggestions: string[] = [];
  const rc = bundle.dataQuality.recordCounts;
  const lowest = Object.entries(rc).sort((a, b) => a[1] - b[1])[0];
  if (lowest) {
    suggestions.push(
      `Log more ${lowest[0]} records. You currently have ${lowest[1]} for this period.`,
    );
  }
  for (const skipped of bundle.dataQuality.skippedAnalyses) {
    suggestions.push(`${skipped.analysis}: ${skipped.reason}`);
  }
  if (suggestions.length === 0) {
    suggestions.push(
      'Keep logging glucose readings, meals, and activity consistently for a few more days.',
    );
  }
  return {
    shortAnswer: "I don't have enough evidence from your data to answer that yet.",
    detail: ['Here is what would help build the evidence:', ...suggestions],
    citedFindingIds: [],
    confidence: 'insufficient',
    suggestedQuestionsForClinician: [],
    notEnoughData: true,
  };
}

const EVIDENCE_PHRASE: Record<Finding['evidenceLevel'], string> = {
  INSUFFICIENT: 'not enough data to rely on',
  EARLY: 'an early signal only',
  EMERGING: 'an emerging pattern',
  CONSISTENT: 'a consistent pattern',
};

function formatFindingLine(f: Finding): string {
  return `${f.statement} ${formatFindingBasis(f)}`;
}

/** The provenance half of a finding, usable on its own when the statement is already the headline. */
function formatFindingBasis(f: Finding): string {
  return `That comes from ${lowerFirst(f.basis.replace(/\.$/, ''))}: ${f.sampleSize} data ${f.sampleSize === 1 ? 'point' : 'points'}, ${EVIDENCE_PHRASE[f.evidenceLevel]}.`;
}

function lowerFirst(text: string): string {
  return text.length === 0 ? text : text[0]!.toLowerCase() + text.slice(1);
}

/** De-duplicate caveats and render them as separate sentences, not one run-on. */
function formatCaveats(findings: readonly Finding[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const finding of findings) {
    for (const caveat of finding.caveats ?? []) {
      const trimmed = caveat.trim().replace(/\.$/, '');
      const key = trimmed.toLowerCase();
      if (trimmed.length === 0 || seen.has(key)) continue;
      seen.add(key);
      out.push(`${trimmed}.`);
    }
  }
  return out;
}

/**
 * Turn a finding into something a person could actually say out loud in an
 * appointment, rather than a restatement of the finding with a prefix.
 */
function clinicianQuestion(f: Finding): string {
  const subject = lowerFirst(f.statement.replace(/^Your /, '').replace(/\.$/, ''));
  return `My tracking shows ${subject}. Does that match what you would expect, and is it worth doing anything about?`;
}

function buildAssistantAnswer(question: string, bundle: EvidenceBundle): unknown {
  const relevant = pickRelevantFindings(question, bundle.findings);
  const usable = relevant.filter((f) => f.evidenceLevel !== 'INSUFFICIENT');

  if (usable.length === 0) {
    return buildNotEnoughDataAnswer(bundle);
  }

  const top = usable[0]!;
  const shortAnswer =
    top.statement.length <= 280 ? top.statement : `${top.statement.slice(0, 277)}...`;

  // The headline already carries the top finding; repeating it verbatim as the
  // first paragraph just makes the answer look padded. Lead with the evidence
  // behind it instead, then any further findings that bear on the question.
  const detail: string[] = [formatFindingBasis(top)];
  for (const finding of usable.slice(1)) detail.push(formatFindingLine(finding));

  const caveats = formatCaveats(usable);
  if (caveats.length > 0) {
    detail.push(`Worth keeping in mind: ${caveats.join(' ')}`);
  }
  detail.push(
    'This reflects what your own data shows. It is not medical advice or a clinical assessment.',
  );

  return {
    shortAnswer,
    detail,
    citedFindingIds: usable.map((f) => f.id),
    confidence: overallConfidence(usable),
    suggestedQuestionsForClinician: usable
      .filter((f) => f.evidenceLevel === 'EMERGING' || f.evidenceLevel === 'CONSISTENT')
      .slice(0, 3)
      .map(clinicianQuestion),
    notEnoughData: false,
  };
}

function buildMealParse(question: string): unknown {
  // The local provider does not run an NLP model; it cannot reliably parse
  // free text into structured nutrition estimates without one. It always
  // returns the input as unparsed rather than fabricating numbers.
  return {
    meals: [],
    exercise: [],
    unparsed: [question],
  };
}

function buildWeeklyNarrative(bundle: EvidenceBundle): unknown {
  const usable = bundle.findings.filter((f) => f.evidenceLevel !== 'INSUFFICIENT');
  const sorted = [...usable].sort(
    (a, b) => EVIDENCE_RANK[b.evidenceLevel] - EVIDENCE_RANK[a.evidenceLevel],
  );

  if (sorted.length === 0) {
    return {
      headline: 'Not enough data yet for a weekly summary',
      sections: [],
      whatChanged: 'Not enough data yet to say what changed.',
      whatWentWell: 'Not enough data yet to highlight what went well.',
      whatToExploreNext: 'Keep logging glucose readings, meals, and activity consistently.',
      questionsForClinician: [],
    };
  }

  return {
    headline: sorted[0]!.statement,
    sections: sorted.slice(0, 5).map((f) => ({ heading: f.kind, body: formatFindingLine(f) })),
    whatChanged: sorted[0]?.statement ?? 'No clear change identified this period.',
    whatWentWell:
      sorted.find((f) => f.evidenceLevel === 'CONSISTENT')?.statement ??
      'Keep up your current logging habits.',
    whatToExploreNext:
      'Discuss the patterns above with your healthcare provider before making any changes.',
    questionsForClinician: sorted
      .filter((f) => f.evidenceLevel === 'EMERGING' || f.evidenceLevel === 'CONSISTENT')
      .slice(0, 3)
      .map((f) => `A pattern worth discussing with your healthcare provider: ${f.statement}`),
  };
}

function buildNoteStructure(question: string): unknown {
  // Same reasoning as buildMealParse: without a model, don't guess.
  return {
    candidates: [],
    unparsed: [question],
  };
}

/** Best-effort task detection from the last user message + presence of a matching schema shape. */
function detectTask(req: CompletionRequest): 'answer' | 'meal' | 'narrative' | 'note' {
  const schema = req.responseSchema as { properties?: Record<string, unknown> };
  const props = schema.properties ? Object.keys(schema.properties) : [];
  if (props.includes('meals') && props.includes('exercise')) return 'meal';
  if (props.includes('sections') && props.includes('headline')) return 'narrative';
  if (props.includes('candidates')) return 'note';
  return 'answer';
}

function lastUserMessage(req: CompletionRequest): string {
  for (let i = req.messages.length - 1; i >= 0; i -= 1) {
    const m = req.messages[i];
    if (m && m.role === 'user') return m.content;
  }
  return '';
}

/**
 * Extracts the evidence bundle that the caller embedded in the prompt.
 * `lib/ai/pipeline.ts` always JSON-serializes the bundle into the system
 * prompt under a fixed marker so the local provider can recover it without
 * a second parameter — this keeps the `AIProvider` interface identical
 * across all providers.
 */
function extractBundle(req: CompletionRequest): EvidenceBundle | null {
  const marker = '<<EVIDENCE_BUNDLE_JSON>>';
  const idx = req.system.indexOf(marker);
  if (idx === -1) return null;
  const rest = req.system.slice(idx + marker.length);
  const endMarker = '<<END_EVIDENCE_BUNDLE_JSON>>';
  const endIdx = rest.indexOf(endMarker);
  const jsonText = endIdx === -1 ? rest : rest.slice(0, endIdx);
  try {
    return JSON.parse(jsonText) as EvidenceBundle;
  } catch {
    return null;
  }
}

export class LocalProvider implements AIProvider {
  readonly id = 'local';
  readonly name = 'On-server (no external AI)';
  readonly isExternal = false;

  available(): boolean {
    return true;
  }

  async complete(req: CompletionRequest): Promise<CompletionResult> {
    const bundle = extractBundle(req);
    const task = detectTask(req);
    const question = lastUserMessage(req);

    let json: unknown;
    if (!bundle) {
      // No bundle could be recovered — safest possible fallback.
      json = {
        shortAnswer: "I don't have enough evidence from your data to answer that yet.",
        detail: ["I don't have enough evidence from your data to answer that yet."],
        citedFindingIds: [],
        confidence: 'insufficient' satisfies ConfidenceLevel,
        suggestedQuestionsForClinician: [],
        notEnoughData: true,
      };
    } else if (task === 'meal') {
      json = buildMealParse(question);
    } else if (task === 'narrative') {
      json = buildWeeklyNarrative(bundle);
    } else if (task === 'note') {
      json = buildNoteStructure(question);
    } else {
      json = buildAssistantAnswer(question, bundle);
    }

    return { json, providerId: this.id };
  }
}

// Re-exported for tests that want to exercise the relevance ranking directly
// without going through the full `complete()` marker-extraction plumbing.
export const __internal = { pickRelevantFindings, overallConfidence, tokenize };
