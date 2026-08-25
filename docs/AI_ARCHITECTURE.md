# AI architecture

DiaLog's AI assistant is built so that the model never sees a health record — only pre-aggregated, evidence-graded statistics — and every response it produces is filtered through deterministic guardrails before a user sees it. This document describes that pipeline precisely, citing the files that implement each piece.

## The analytics → evidence → AI separation

```
 raw records (Prisma)
        │
        ▼
 lib/analytics/engine.ts  runAnalytics()      ← owned independently of lib/ai; AI never imports this module's types
        │  AnalyticsResult (summary, graded Finding[], anomalies, trend, clusters, feature importance)
        ▼
 lib/services/analytics-service.ts  toEvidenceBundle()
        │  EvidenceBundle  (lib/ai/types.ts — the ONLY shape lib/ai is allowed to reason from)
        ▼
 lib/ai/redact.ts  redactForProvider()        ← strips free-text-shaped fields for external providers without consent
        │
        ▼
 lib/ai/pipeline.ts  assertNoRawRecords()  →  provider.complete()  →  lib/ai/guardrails.ts
        │
        ▼
 AssistantAnswer / WeeklyNarrative / MealParse / NoteStructure  →  rendered to the user
```

`lib/ai/types.ts` states the boundary directly: _"The analytics engine produces its own `AnalyticsResult` shape. The AI layer never imports it — the app layer is responsible for mapping analytics output onto this `EvidenceBundle` shape before calling into `lib/ai`."_ This is enforced two ways, not just documented:

1. **Structurally** — `lib/ai/**` has no import of anything under `lib/analytics/**`; the only bridge is `lib/services/analytics-service.ts`.
2. **At runtime** — `assertNoRawRecords()` (`lib/ai/pipeline.ts`) walks the bundle before every AI call and throws if `summary` contains a non-scalar value, if `findings` contains anything that isn't `Finding`-shaped, or if a finding's `metrics` contains an array/object — i.e. if a raw record array ever leaked into the bundle by mistake, the call fails loudly instead of silently sending it.

## What the LLM does and does not receive

The `EvidenceBundle` (`lib/ai/types.ts`) is the entire contract:

```ts
export interface EvidenceBundle {
  generatedAt: string;
  periodStart: string;
  periodEnd: string;
  units: 'mg/dL' | 'mmol/L';
  targetRange: { low: number; high: number };
  summary: Record<string, number | string | null>; // scalars only
  findings: Finding[]; // graded, pre-computed statements
  dataQuality: {
    recordCounts: Record<string, number>;
    coverageDays: number;
    skippedAnalyses: { analysis: string; reason: string }[];
  };
}
```

and each `Finding` (`lib/domain/evidence.ts`):

```ts
export interface Finding {
  id: string;
  kind: string;
  statement: string; // neutral, factual statement of what the data shows
  sampleSize: number;
  evidenceLevel: EvidenceLevel;
  source: 'STATISTICAL' | 'ML' | 'REFERENCE';
  metrics: Record<string, number | string | null>; // scalars only
  basis: string; // which records were compared, in words
  periodStart: string;
  periodEnd: string;
  caveats?: string[];
}
```

**Never sent**: individual glucose readings, meal descriptions, medication names/doses, symptom or mood notes, or any other raw record. **Sent**: counts, averages, percentages, evidence-graded statements about patterns (e.g. "your post-dinner readings are higher on days without an evening walk"), and which analyses were skipped and why. `toEvidenceBundle()` (`lib/services/analytics-service.ts`) is the single function that constructs this from an `AnalyticsResult` — it maps scalar summary numbers (already unit-converted for display) and passes `result.findings` through unchanged. Each finding's `statement` text is already worded in the user's chosen unit by that point: `analyzeUser()` passes `profile.glucoseUnit` into the analytics engine as `AnalyticsInput.displayUnit`, and every finding-building function in `lib/analytics/*` renders its numbers through `lib/analytics/format.ts` (`formatLevel()`/`formatDelta()`), never mg/dL text hardcoded, so an mmol/L user reads mmol/L in an assistant answer too.

`lib/ai/redact.ts` adds a second, belt-and-suspenders layer specifically for external providers: if the selected provider `isExternal` (Anthropic/OpenAI) and the user has not set `externalAiConsentAt`, `redactForProvider()` nulls out any `summary` key that looks like it carries free text (matching `/note|comment|freetext|free_text|diary|journal/i`) and strips any `Finding.caveats` entry that looks like a quoted string. The comment in that file is explicit that this is defensive, not the primary control — the primary control is that nothing in the current `EvidenceBundle`/`Finding` shape carries verbatim personal free text in the first place.

## The provider abstraction

`lib/ai/provider.ts` defines one interface every provider implements identically:

```ts
export interface AIProvider {
  id: string;
  name: string;
  isExternal: boolean; // true when data leaves this deployment
  available(): boolean;
  complete(req: CompletionRequest): Promise<CompletionResult>;
}
```

Three concrete providers (`lib/ai/providers/`):

| Provider                     | `isExternal` | `available()`         | Mechanism                                                                                                                                                                                                                                           |
| ---------------------------- | ------------ | --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `local` (`local.ts`)         | `false`      | always `true`         | No network call, ever. Deterministic templates over `Finding.kind`/`statement`/`basis` with simple keyword-overlap relevance ranking (`relevanceScore`, `pickRelevantFindings`).                                                                    |
| `anthropic` (`anthropic.ts`) | `true`       | `!!ANTHROPIC_API_KEY` | Plain `fetch` to `api.anthropic.com/v1/messages` (no SDK dependency), forcing structured output via a single tool whose `input_schema` is the caller's JSON Schema and `tool_choice` pinned to it. 30s timeout. Never logs request/response bodies. |
| `openai` (`openai.ts`)       | `true`       | `!!OPENAI_API_KEY`    | Equivalent `fetch`-based implementation against the OpenAI API.                                                                                                                                                                                     |

`getProvider(preferred?)` resolution order: explicit `preferred` argument → `AI_PROVIDER` env var → `'local'`. Whatever is selected, if `available()` is false the call transparently falls back to `local` — so an unconfigured `AI_PROVIDER=anthropic` deployment still works, just without a hosted model. This means **the product works completely with zero API keys.**

### How the local provider works

`LocalProvider.complete()` (`lib/ai/providers/local.ts`) recovers the `EvidenceBundle` the caller embedded as JSON inside the system prompt (between fixed `<<EVIDENCE_BUNDLE_JSON>>` / `<<END_EVIDENCE_BUNDLE_JSON>>` markers — see `extractBundle()`), so it needs no second parameter and satisfies the exact same `AIProvider` interface as a network provider. It then:

- Detects which structured-output task is being asked for by inspecting the response schema's declared property names (`detectTask()`).
- For **answer-a-question**: scores every finding for keyword overlap with the question (`tokenize()` + a small stopword list), keeps findings with `evidenceLevel !== 'INSUFFICIENT'`, and composes an answer from the highest-scoring ones — or returns the explicit "not enough evidence" shape (`buildNotEnoughDataAnswer`) if nothing usable matched, complete with concrete suggestions drawn from `dataQuality.skippedAnalyses`.
- For **meal/note parsing**: honestly returns everything as `unparsed` — the local provider has no NLP model and, per its own code comment, "cannot reliably parse free text into structured nutrition estimates without one," so it never fabricates numbers.
- For **weekly narrative**: assembles headline/sections/what-changed from the highest-evidence findings, or the explicit "not enough data yet" narrative if none qualify.

## Structured output schemas

Every task has a paired Zod schema (runtime validation) and hand-written JSON Schema (sent to LLM providers to constrain generation) in `lib/ai/schemas.ts` — kept in sync by hand and cross-checked by fixture tests, per that file's own header comment, since Zod doesn't currently export a lossless matching JSON Schema:

| Task              | Zod type                | Key fields                                                                                                                                                                    |
| ----------------- | ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `AssistantAnswer` | `AssistantAnswerSchema` | `shortAnswer` (≤280 chars), `detail[]`, `citedFindingIds[]`, `confidence` (`insufficient\|low\|moderate\|high`), `suggestedQuestionsForClinician[]`, `notEnoughData: boolean` |
| `MealParse`       | `MealParseSchema`       | `meals[]` / `exercise[]`, each field explicitly named `estimated*` and carrying a `confidence` (`low\|medium\|high`); `unparsed[]` for anything not confidently extracted     |
| `WeeklyNarrative` | `WeeklyNarrativeSchema` | `headline`, `sections[]`, `whatChanged`, `whatWentWell`, `whatToExploreNext`, `questionsForClinician[]`                                                                       |
| `NoteStructure`   | `NoteStructureSchema`   | `candidates[]` (typed `glucose\|meal\|exercise\|medication\|symptom\|note`, free-form `fields`, `confidence`), `unparsed[]`                                                   |

## Guardrails (`lib/ai/guardrails.ts`)

Every function here is pure — no I/O, no provider calls — so it is fully unit-testable, and `lib/ai/pipeline.ts` is the only production caller.

| Guardrail                                                           | What it catches                                                                                                                                                                                                                                                                                                                                                                                                                  | Behaviour on failure                                                                                                                                                                                                                                                             |
| ------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Schema validation** (`validateAssistantAnswer`)                   | Malformed/incomplete JSON from the model that doesn't match `AssistantAnswerSchema`.                                                                                                                                                                                                                                                                                                                                             | Rejects the response outright → falls back to `safeFallbackAnswer()`.                                                                                                                                                                                                            |
| **Medical-safety filter** (`checkMedicalSafety`, `UNSAFE_PATTERNS`) | Seven deliberately over-matching regex families across every text field of the answer: **dosing-instruction** (increase/decrease/adjust/stop/start + insulin/dose/medication/metformin/etc.), **take-units** ("take 5 units"), **diagnosis** ("you have diabetes"/"you are diabetic"), **diagnose-verb** (any form of "diagnose"), **skip-doctor** ("you don't need to see a doctor"), **clinically-proven**, **medical-grade**. | Rejects the response → falls back to `safeFallbackAnswer()`. The code comment states the philosophy directly: a false rejection just falls back to a safe templated answer, while a false negative could ship unsafe medical advice — so the patterns are written to over-match. |
| **Grounding check** (`enforceGrounding`)                            | Any `citedFindingIds` entry that doesn't correspond to a real finding in the bundle.                                                                                                                                                                                                                                                                                                                                             | Drops the unknown ids. If none remain and `notEnoughData` wasn't already true, downgrades `confidence` to `'low'` and appends the standard caveat: _"This is based on limited evidence — treat it as a first hint, not a conclusion."_                                           |
| **Claim consistency** (`enforceClaimConsistency`)                   | A model claiming useful confidence when every finding in the bundle is `INSUFFICIENT` (or there are no findings at all).                                                                                                                                                                                                                                                                                                         | Forces `notEnoughData: true`, `confidence: 'insufficient'`, `citedFindingIds: []`, regardless of what the model said.                                                                                                                                                            |

`applyAssistantAnswerGuardrails()` runs these in order (schema → safety → grounding → claim consistency) and returns `{ answer, rejected, notes }`. In `lib/ai/pipeline.ts`, a `rejected: true` result from a **live** provider triggers exactly one retry against `LocalProvider`; if that also fails guardrails, the user gets `safeFallbackAnswer()` — the assistant never has zero fallback path.

## Evidence-grading thresholds (`lib/domain/evidence.ts`)

Every analytical statement is graded by how many observations support it, using per-analysis-type minimum sample sizes, deliberately conservative so "a personal pattern claimed from a handful of readings is not a pattern" (the file's own words):

| Analysis type                                   | INSUFFICIENT (below) | EARLY | EMERGING | CONSISTENT |
| ----------------------------------------------- | -------------------- | ----- | -------- | ---------- |
| `summary` (e.g. "your average morning reading") | < 5                  | 5–13  | 14–29    | ≥ 30       |
| `comparison` (two groups of days/readings)      | < 8                  | 8–19  | 20–39    | ≥ 40       |
| `association` (behaviour ↔ glucose outcome)    | < 10                 | 10–23 | 24–49    | ≥ 50       |
| `trend` (change over time)                      | < 10                 | 10–20 | 21–44    | ≥ 45       |
| `model` (personalised model fit)                | < 30                 | 30–79 | 80–149   | ≥ 150      |

`gradeEvidence(sampleSize, analysisKind)` returns the `EvidenceLevel` enum value; `EVIDENCE_LABELS` supplies the plain-language label/description shown in the UI (e.g. `EARLY` → "Early signal — Based on a small number of records. Treat this as a first hint, not a conclusion."). Every `Finding` the analytics engine produces carries one of these grades, and it is this grade — not the LLM's own judgement — that ultimately gates whether the guardrails let a claim stand.

## "Not enough data" behaviour

This is a first-class outcome, not an error path:

- `lib/analytics/engine.ts`'s `runAnalytics()` explicitly records `skippedAnalyses` with a human-readable reason whenever trend detection, day-pattern clustering, feature importance, or any of the five association kinds (`post-meal-carb-bucket`, `post-dinner-activity`, `sleep-duration`, `fasting-weekday-weekend`, `stress`) don't have enough paired data — e.g. _"Fewer than 10 days with a glucose reading (have 4)."_
- The local provider's `buildNotEnoughDataAnswer()` turns those skip reasons directly into actionable suggestions ("Log more meal records — you currently have 3 for this period").
- `enforceClaimConsistency()` guarantees that even a live LLM cannot override this: if every finding is `INSUFFICIENT`, the final answer is forced to `notEnoughData: true` regardless of what the model produced.
- The weekly narrative and meal/note parsers each have their own explicit "not enough data yet" / "everything unparsed" shape rather than a generic error, so the UI always has something honest to render.
