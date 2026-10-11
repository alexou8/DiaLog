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
 lib/ai/pipeline.ts
   detectEmergency(question)       ← lib/ai/emergency.ts; a possible emergency gets a fixed answer, no model call
   assertNoRawRecords(bundle)
   enforceExternalConsent()        ← lib/ai/provider.ts; external provider + no consent → local provider
   redactForProvider()             ← lib/ai/redact.ts; defence in depth behind the consent gate
   callProvider() → provider.complete()
   guardrails (lib/ai/guardrails.ts) on every task's output → local provider → fixed safe value
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

The bundle is not the only thing a request carries: the user's **question**, and the **quick-log or note text** for the parsing tasks, are sent as the user message, word for word. That free text is why the consent gate below exists.

## The consent gate

An external provider (`isExternal: true`: Anthropic, OpenAI) is used only for a user who has set `externalAiConsentAt`. Without consent every task runs on the local provider and nothing leaves the server: not the bundle, not the question, not quick-log text.

- `enforceExternalConsent(provider, externalConsent)` (`lib/ai/provider.ts`) returns the local provider when `provider.isExternal && !externalConsent`. `resolveProviderForUser({ preferred, externalConsent })` is `getProvider()` followed by that gate, and is what `lib/actions/assistant.ts` uses.
- Every pipeline entry point (`answerQuestion`, `parseMealText`, `summarizePeriod`, `parseNoteText`) takes a **required** `externalConsent` argument and applies the gate itself, so a caller cannot skip it and the compiler rejects one that forgets it. Every request then goes through `callProvider()`, which throws a plain `Error` (never swallowed by the local fallback) if an external provider is reached without consent.
- Why it lives in the pipeline: before 2026-10 the only consent check was a `redactForProvider()` call in the server action. It stripped free-text bundle fields (which the bundle never contains), while the question went to the third party verbatim; quick-log called `parseMealText` with no consent check at all. Both broke the promise on the settings page.
- `lib/ai/redact.ts` stays on the path inside the pipeline as defence in depth. With the gate in front of it, it is a no-op today; it exists so that loosening the gate later cannot on its own put free-text bundle fields in front of an external provider. It nulls any `summary` key that looks like free text (`/note|comment|freetext|free_text|diary|journal/i`) and drops quoted-looking `Finding.caveats`.

Tests: `tests/unit/ai/assistant-consent.test.ts` runs both server actions with `AI_PROVIDER=anthropic` and a key configured, and asserts `fetch` is never called without consent and is called with it; `tests/unit/ai/pipeline.test.ts` asserts the same for all four pipeline tasks.

## The emergency check (`lib/ai/emergency.ts`)

`answerQuestion` scans the question before anything else, including the bundle check. A question describing a possible emergency gets a fixed answer (`emergencyAnswer()`) and no provider is called:

- **Signals**: a glucose number below 54 mg/dL or 3.0 mmol/L (a bare number counts only with glucose wording just before it and nothing after it that makes it a time, dose or carb count), a meter reading "LO", and phrases such as passing out, unconscious, seizure, can't wake, DKA, high ketones, chest pain, vomiting and can't keep anything down, overdose or too much insulin, and self-harm. A small French set is included. Text is normalised with the guardrails' `normalizeForSafety()` first.
- **The answer** tells the person to follow their care team's plan for a low, to call emergency services for someone unconscious, seizing or unrousable, and to contact emergency services or their care team for high ketones, vomiting or chest pain. It deliberately contains no amounts or dosing: what to do for a low is in their care plan, and a number from DiaLog is exactly what the guardrails exist to stop.
- Why: before 2026-10 "My sugar is 45 and I'm shaking, what do I do?" went down the normal path and got either keyword-matched findings or "not enough evidence". Like the guardrails, the check over-matches on purpose.
- The server action returns `emergency: true` with the answer so the UI can treat it differently.
- `parseMealText` (quick log) runs the same check on the logged text. On a hit it parses with the local engine only, so nothing waits on or goes to a model, and returns `emergency: true` and `emergencyAnswer` alongside the proposal. That way "BG 2.8 and shaking, had juice" still offers the juice to save, with the safety template shown first.

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

| Provider                     | `isExternal` | `available()`         | Mechanism                                                                                                                                                                                                                                                                                                                                                                          |
| ---------------------------- | ------------ | --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `local` (`local.ts`)         | `false`      | always `true`         | No network call, ever. Deterministic templates over `Finding.kind`/`statement`/`basis` with simple keyword-overlap relevance ranking (`relevanceScore`, `pickRelevantFindings`).                                                                                                                                                                                                   |
| `anthropic` (`anthropic.ts`) | `true`       | `!!ANTHROPIC_API_KEY` | Plain `fetch` to `api.anthropic.com/v1/messages` (no SDK dependency). Default model `claude-opus-5-5` (`ANTHROPIC_MODEL` overrides). Structured outputs via `output_config.format` (`json_schema`), `effort: 'low'`, and `fallbacks: 'default'` with the `server-side-fallback-2026-07-01` beta on models that support it. No `temperature`, `tools`, `tool_choice` or `thinking`. |
| `openai` (`openai.ts`)       | `true`       | `!!OPENAI_API_KEY`    | Equivalent `fetch`-based implementation against Chat Completions, with `response_format` `json_schema` in strict mode and `max_completion_tokens`.                                                                                                                                                                                                                                 |

Both hosted providers share `providers/http.ts`: one 30s timeout that covers the response **body** as well as the headers (it used to be cleared when headers arrived, so a stalled body hung the server action), logs of status, duration and the API's error `type` only, and errors that never carry a body. `providers/schema-compat.ts` strips the JSON Schema keywords each provider's structured-output mode rejects (Anthropic: numeric, length and item-count limits and `pattern`; OpenAI strict: `minLength`/`maxLength`); zod still enforces every cap on the way back.

Why the Anthropic request changed (2026-10): the old default `claude-sonnet-5` rejects a non-default `temperature`, and current models reject forced `tool_choice`, both with a 400. Every request failed and the pipeline silently answered from the local engine. `tests/unit/ai/anthropic-provider.test.ts` now asserts the exact request body.

`getProvider(preferred?)` resolution order: explicit `preferred` argument → `AI_PROVIDER` env var → `'local'`. Whatever is selected, if `available()` is false the call transparently falls back to `local` — so an unconfigured `AI_PROVIDER=anthropic` deployment still works, just without a hosted model. This means **the product works completely with zero API keys.**

### How the local provider works

`LocalProvider.complete()` (`lib/ai/providers/local.ts`) recovers the `EvidenceBundle` the caller embedded as JSON inside the system prompt (between fixed `<<EVIDENCE_BUNDLE_JSON>>` / `<<END_EVIDENCE_BUNDLE_JSON>>` markers — see `extractBundle()`), so it needs no second parameter and satisfies the exact same `AIProvider` interface as a network provider. It then:

- Detects which structured-output task is being asked for by inspecting the response schema's declared property names (`detectTask()`). Meal and note parsing are dispatched before the bundle lookup, because those prompts carry no bundle; they used to fall into the "no bundle" branch, fail validation and be called twice, so every local quick-log reported a fallback.
- For **answer-a-question**: scores every finding for keyword overlap with the question (`tokenize()`, which keeps letters in any script, + a small stopword list), keeps findings with `evidenceLevel !== 'INSUFFICIENT'`, and composes an answer from the highest-scoring ones — or returns the explicit "not enough evidence" shape (`buildNotEnoughDataAnswer`) if nothing usable matched, complete with concrete suggestions drawn from `dataQuality.skippedAnalyses`. Keyword matching only understands English, so a question that looks non-English (`looksNonEnglish()`: non-Latin letters, or two common French/Spanish/Portuguese/German function words) gets a general summary of the strongest findings, introduced as such, instead of a misleading "not enough evidence".
- For **meal/note parsing**: honestly returns everything as `unparsed` — the local provider has no NLP model and, per its own code comment, "cannot reliably parse free text into structured nutrition estimates without one," so it never fabricates numbers.
- For **weekly narrative**: assembles headline/sections/what-changed from the highest-evidence findings, or the explicit "not enough data yet" narrative if none qualify.

## Structured output schemas

Every task has a paired Zod schema (runtime validation) and hand-written JSON Schema (sent to LLM providers to constrain generation) in `lib/ai/schemas.ts` — kept in sync by hand and cross-checked by fixture tests, per that file's own header comment, since Zod doesn't currently export a lossless matching JSON Schema:

| Task              | Zod type                | Key fields                                                                                                                                                                    |
| ----------------- | ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `AssistantAnswer` | `AssistantAnswerSchema` | `shortAnswer` (≤280 chars), `detail[]`, `citedFindingIds[]`, `confidence` (`insufficient\|low\|moderate\|high`), `suggestedQuestionsForClinician[]`, `notEnoughData: boolean` |
| `MealParse`       | `MealParseSchema`       | `meals[]` / `exercise[]`, each field explicitly named `estimated*` and carrying a `confidence` (`low\|medium\|high`); `unparsed[]` for anything not confidently extracted     |
| `WeeklyNarrative` | `WeeklyNarrativeSchema` | `headline`, `sections[]`, `whatChanged`, `whatWentWell`, `whatToExploreNext`, `questionsForClinician[]`                                                                       |
| `NoteStructure`   | `NoteStructureSchema`   | `candidates[]` (typed `glucose\|meal\|exercise\|medication\|symptom\|note`, `fields` as a list of `{ key, value }`, `confidence`), `unparsed[]`                               |

Every string and array is capped (`LIMITS` in `schemas.ts`) and `shortAnswer` must be non-empty; an empty answer and unbounded lists used to validate. Every object is closed (`additionalProperties: false`) with every property required, which OpenAI strict mode demands: `NoteStructure.fields` used to be a free-form record (`additionalProperties: {...}`), a guaranteed 400 from both providers. `tests/unit/ai/schema-compat.test.ts` checks this for all four schemas.

## Guardrails (`lib/ai/guardrails.ts`)

Every function here is pure — no I/O, no provider calls — so it is fully unit-testable, and `lib/ai/pipeline.ts` is the only production caller.

| Guardrail                                                                                 | What it catches                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | Behaviour on failure                                                                                                                                                                                                                   |
| ----------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Schema validation** (`validateAssistantAnswer`)                                         | Malformed/incomplete JSON from the model that doesn't match `AssistantAnswerSchema`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Rejects the response outright → falls back to `safeFallbackAnswer()`.                                                                                                                                                                  |
| **Medical-safety filter** (`checkMedicalSafety`, `findUnsafeLanguage`, `UNSAFE_PATTERNS`) | Deliberately over-matching regex families, run over `normalizeForSafety()` text: **dosing-instruction** (an action verb such as increase, skip, hold, double, cut, bump, inject, take, or French/Spanish equivalents, near insulin, a brand or generic diabetes medication, dose or units), **dose-quantity** (any number, digits or words, next to a dose unit: "four units", "4 U", "0.5 mg", "2 unités"), **medication-quantity** ("Lantus 20"), **insulin-delivery-term** (inject, bolus, basal), **ratio-or-correction** (insulin-to-carb, "1 unit per 10 g", "1:10" near a ratio word, correction or sensitivity factor), **treatment-instruction** ("Eat 15 g of fast carbs"), **diagnosis** ("you're diabetic", "indicates type 2 diabetes"), **diagnose-verb**, **skip-doctor** ("you don't need a doctor", "no need to see a doctor"), **clinically-proven**, **medical-grade**, and **causal-claim** (cause, leads to, because of, due to, drives, triggers, makes your... with a glucose word in the same sentence; negated claims and questions are allowed). | Rejects the response → local provider → `safeFallbackAnswer()`. A false rejection just falls back to a safe templated answer, while a false negative could ship unsafe medical advice — so the patterns are written to over-match.     |
| **Confidence cap** (`enforceConfidenceCap`)                                               | A model claiming more confidence than its evidence supports, or citing an `INSUFFICIENT` finding as support.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Drops `INSUFFICIENT` citations and clamps `confidence` to the best cited grade (`CONSISTENT`→high, `EMERGING`→moderate, `EARLY`→low). With nothing left to cite, confidence becomes `low` and the limited-evidence caveat is added.    |
| **Grounding check** (`enforceGrounding`)                                                  | Any `citedFindingIds` entry that doesn't correspond to a real finding in the bundle.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Drops the unknown ids. If none remain and `notEnoughData` wasn't already true, downgrades `confidence` to `'low'` and appends the standard caveat: _"This is based on limited evidence — treat it as a first hint, not a conclusion."_ |
| **Claim consistency** (`enforceClaimConsistency`)                                         | A model claiming useful confidence when every finding in the bundle is `INSUFFICIENT` (or there are no findings at all).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | Forces `notEnoughData: true`, `confidence: 'insufficient'`, `citedFindingIds: []`, regardless of what the model said.                                                                                                                  |

`applyAssistantAnswerGuardrails()` runs these in order (schema → safety → grounding → confidence cap → claim consistency) and returns `{ answer, rejected, notes }`. In `lib/ai/pipeline.ts`, every task goes through `runTask()`: a provider error or a rejected output from a **live** provider triggers exactly one attempt against `LocalProvider`; if that also fails, the user gets a fixed safe value (`safeFallbackAnswer()`, the "not enough data" narrative, or the input returned as `unparsed`). The local provider is never called twice.

### Normalisation and scope

- **Normalisation** (`normalizeForSafety()`): NFKC, then strip `\p{Cf}` (zero-width characters), map every Unicode decimal digit to ASCII, drop combining marks, fold curly apostrophes and dashes, lower-case. Each step closes a bypass from the 2026-10 review, in which 24 of 27 constructed unsafe outputs ("Take ٤ units", "Take ４ units", "in​sulin", "Take four units of Humalog", "Augmentez votre insuline de 2 unités", ...) passed the old filter. All 27 are now must-catch cases in `tests/unit/ai/guardrails-bypass.test.ts`, next to ordinary data phrasing that must pass.
- **Scope**: the filter runs on every task's output, not just answers. `parseMealText` and `parseNoteText` scan every string of the validated result; a string that is the user's whole input, or one whole sentence or clause of it, is exempt (it is their words echoed back, not advice). Anything else is checked, including a fragment cut out of a clause: "take 10 units" lifted from "don't take 10 units" is not an echo. A note's `medication` candidate is also rejected if it carries a number the user never wrote. `summarizePeriod` scans every string of the narrative. Before 2026-10 only `answerQuestion` was filtered.
- **Language**: the patterns are English plus a small French set (unités, injecter, prendre X) and a few Spanish words. Output in other languages is a known gap; see Limitations.

### Limitations

- Numbers in model prose are not checked against the bundle, so an invented "your average is 182" in `detail` is not caught (only confidence is capped).
- Output in a language other than English, French or (partly) Spanish is not reliably filtered.
- The trend label for a significantly falling mean is still `improving` in `lib/analytics/ml/trend.ts`, because renaming it needs coordinated changes in `lib/analytics/insights.ts` and `app/app/reports/page.tsx`. Until then, `lib/ai/prompts.ts` rewrites it to `falling` before the bundle reaches a model, and the system prompt forbids "improving"/"worsening".

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

**Two-group comparisons are graded on the smaller arm.** `gradeComparisonEvidence()` grades `2 × min(nA, nB)` against the table, not `nA + nB`: 38 vs 2 is as weak as 2 vs 2. It also caps the grade at `EARLY` unless the difference is both statistically distinguishable (p < 0.05) and not negligible (|Cohen's d| ≥ 0.2). An analysis whose comparison grades `INSUFFICIENT` on that rule returns no finding, so 30 short-sleep nights against 3 adequate ones say nothing. The exception is the post-meal carb-bucket finding: it is a description of every bucket that passes its own `association` gate on the total meal count, and the high-vs-low comparison only sets its grade. A comparison too small to grade leaves it at `EARLY` rather than removing it, so logging more meals can never make an existing finding disappear.

## "Not enough data" behaviour

This is a first-class outcome, not an error path:

- `lib/analytics/engine.ts`'s `runAnalytics()` explicitly records `skippedAnalyses` with a human-readable reason whenever trend detection, day-pattern clustering, feature importance, or any of the five association kinds (`post-meal-carb-bucket`, `post-dinner-activity`, `sleep-duration`, `fasting-weekday-weekend`, `stress`) don't have enough paired data — e.g. _"Fewer than 10 days with a glucose reading (have 4)."_
- The local provider's `buildNotEnoughDataAnswer()` turns those skip reasons directly into actionable suggestions ("Log more meal records — you currently have 3 for this period").
- `enforceClaimConsistency()` guarantees that even a live LLM cannot override this: if every finding is `INSUFFICIENT`, the final answer is forced to `notEnoughData: true` regardless of what the model produced.
- The weekly narrative and meal/note parsers each have their own explicit "not enough data yet" / "everything unparsed" shape rather than a generic error, so the UI always has something honest to render.
