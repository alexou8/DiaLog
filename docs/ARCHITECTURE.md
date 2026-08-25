# Architecture

This document describes how a request actually moves through DiaLog, layer by layer, citing the real files that implement each concern.

## Layers

```
 ┌─────────────────────────────────────────────────────────────────┐
 │ app/**  (Next.js App Router: Server Components + Server Actions) │
 │   - route files render pages, call services directly             │
 │   - 'use server' actions in lib/actions/** handle mutations       │
 └───────────────────────────────┬───────────────────────────────────┘
                                  │
 ┌────────────────────────────────▼───────────────────────────────────┐
 │ lib/services/**  (application seams)                                 │
 │   analytics-service.ts · import-service.ts · export-service.ts       │
 └───┬─────────────────────────┬─────────────────────────┬─────────────┘
     │                         │                         │
 ┌───▼─────────────┐   ┌───────▼───────────┐   ┌──────────▼───────────┐
 │ lib/analytics/**  │   │ lib/ai/**          │   │ lib/import/**         │
 │ (statistical      │   │ (provider          │   │ (parse, connectors,   │
 │  engine, ml/*)     │   │  abstraction,       │   │  dedupe)              │
 │                    │   │  guardrails)        │   │                      │
 └───┬────────────────┘   └───────┬────────────┘   └──────────┬───────────┘
     │                            │                            │
 ┌───▼────────────────────────────▼────────────────────────────▼──────────┐
 │ lib/domain/**  (pure rules: units, thresholds, evidence, dedupe, time)   │
 └───────────────────────────────┬───────────────────────────────────────┘
                                  │
 ┌────────────────────────────────▼───────────────────────────────────┐
 │ lib/db/**  (Prisma client + scoped queries)  →  PostgreSQL           │
 └───────────────────────────────────────────────────────────────────┘
```

- **Routes** (`app/**`) are Server Components by default; they call `lib/auth/current-user.ts`'s `requireUser`/`requireOnboardedUser` for the session guard, then call a service function directly — there is no internal HTTP hop between a page and its data. The only REST-style endpoints are `app/api/export/route.ts` (data export) and `app/api/health/route.ts` (a liveness/readiness probe that checks the process and a `SELECT 1` against the database, exposing no user data — see docs/DEPLOYMENT.md).
- **Server Actions** (`lib/actions/*.ts`, each file starts with `'use server'`) handle every mutation (sign-up, sign-in, record CRUD, import commit, preference updates, assistant questions). They are the sole write path from the browser; forms `action={someAction}` post directly to them without a hand-written API route.
- **Services** (`lib/services/*.ts`) are the seam between the app layer and the analytics/AI/import subsystems. `analytics-service.ts` is the privacy boundary in code: `analyzeUser()` loads raw records and hands them to `runAnalytics()`, and `toEvidenceBundle()` is the only function allowed to build the `EvidenceBundle` the AI layer sees.
- **Domain/analytics/import/ai** are independent, mostly pure TypeScript modules with their own unit tests, each documented further below and in [AI_ARCHITECTURE.md](AI_ARCHITECTURE.md) and [DEVICE_INTEGRATIONS.md](DEVICE_INTEGRATIONS.md).
- **Data layer** (`lib/db/prisma.ts`, `lib/db/health-records.ts`) is a thin, always-`userId`-scoped wrapper around the Prisma client.
- **Edge middleware** (`middleware.ts`) is a fast-rejection guard only — it redirects unauthenticated visitors away from `/app/*` and authenticated visitors away from `/sign-in`/`/sign-up`, but every server component and action independently re-verifies the session and scopes its own queries, per the comment in `middleware.ts`.

## Request flow: adding a glucose reading

1. `app/app/glucose/new/GlucoseForm.tsx` posts to a Server Action in `lib/actions/records.ts`.
2. The action calls `requireOnboardedUser()` (`lib/auth/current-user.ts`) — redirects to `/sign-in` if there is no valid session, or to `/app/onboarding` if the profile isn't complete.
3. `glucoseEntrySchema.safeParse` (`lib/validation.ts`) validates the form payload; on failure the action returns field errors, no database write happens.
4. The entered value (in whatever unit the form used) is converted to canonical mg/dL via `toMgdl()` (`lib/domain/units.ts`) and checked against `isPlausibleGlucose()` bounds.
5. A `dedupeKey` is computed (`lib/domain/dedupe.ts`) from type + minute-truncated timestamp + rounded value, so a duplicate submit is a no-op rather than a second row.
6. `prisma.glucoseReading.create({ data: { userId: user.id, ... } })` writes the row — every write in this codebase includes the `userId` of the current session, never a client-supplied id.
7. The action calls `revalidatePath` for the affected pages and redirects; the dashboard's next Server Component render re-reads via `lib/db/health-records.ts` and shows the new reading, already unit-converted and band-classified (`lib/domain/thresholds.ts`).

No background job runs. The write is fully synchronous within the one request.

## Request flow: asking the assistant a question

1. `app/app/assistant/AssistantPanel.tsx` submits to `askAssistantAction` (`lib/actions/assistant.ts`).
2. `requireOnboardedUser()` gates the request; if `profile.aiEnabled` is false the action returns early with no AI or analytics work done.
3. `rateLimit('ai:<userId>', ...)` (`lib/auth/rate-limit.ts`) enforces `RATE_LIMITS.ai` (30/hour) before anything expensive runs.
4. `assistantQuestionSchema.safeParse` validates the question text and detail level.
5. `analyzeUser(userId, profile, window)` (`lib/services/analytics-service.ts`) loads the last 90 days of records via `loadAnalyticsWindow()` (`lib/db/health-records.ts`) and runs `runAnalytics()` (`lib/analytics/engine.ts`) — glucose summary, associations, anomalies, trend, day-pattern clustering, feature importance — producing an `AnalyticsResult` full of graded `Finding`s.
6. `toEvidenceBundle()` maps that result onto the `EvidenceBundle` shape (`lib/ai/types.ts`) — aggregates and findings only, unit-converted for display, never a raw record array.
7. `getProvider(process.env.AI_PROVIDER)` (`lib/ai/provider.ts`) selects a provider (falls back to `local` if the requested one is unavailable).
8. `redactForProvider()` (`lib/ai/redact.ts`) strips anything free-text-shaped from the bundle unless the provider is non-external or the user has given `externalAiConsentAt` consent.
9. `answerQuestion()` (`lib/ai/pipeline.ts`) calls `assertNoRawRecords(bundle)` as a runtime defensive check, then `provider.complete()`, then runs the result through `applyAssistantAnswerGuardrails()` (`lib/ai/guardrails.ts`): schema validation → medical-safety regex filter → grounding check (drop citations to findings not in the bundle) → claim-consistency check (force `notEnoughData: true` if every finding is INSUFFICIENT). Any provider failure or guardrail rejection falls back to `LocalProvider`, which is deterministic and always available.
10. The action stores the question and answer as an `AIConversation`/`AIMessage` pair (with the cited evidence attached, so an old answer stays auditable), writes an `AuditEvent`, and returns the answer to the client for render.

## The normalised health-event model

Every health record type (`GlucoseReading`, `Meal`, `ExerciseSession`, `SleepSession`, `MedicationEvent`, `WeightMeasurement`, `BloodPressureMeasurement`, `HydrationEvent`, `SymptomEntry`, `MoodEntry`, `NoteEntry` — see `prisma/schema.prisma`) shares the same shape of concern, even though each has its own table and columns:

- `userId` + `takenAt`, indexed together on every table, because every real query is "this user's records in a time window."
- `source: DataSource` (`MANUAL | IMPORT | AI_ASSISTED | DEVICE | SEED`) — **provenance**, always. The product's evidence-grading and audit story depends on being able to say where a number came from; a raw file import (`IMPORT`) and a user typing a value in (`MANUAL`) are never conflated.
- `dedupeKey: String` with a `@@unique([userId, dedupeKey])` constraint — every record kind is idempotent under re-import. `lib/domain/dedupe.ts` computes it as a SHA-256 hash of `type | externalId | minute-truncated-timestamp | rounded-value | discriminator`, so the same reading exported twice (with float-formatting differences between a CSV and an XML export, say) still collides to the same key. This is why `commitImport()` (`lib/services/import-service.ts`) can safely `createMany({ skipDuplicates: true })`.
- `rawPayload: Json?` on the record types imports actually populate — the original row/element is kept so provenance is always answerable, without polluting the typed columns other code reads.

## Canonical units

Glucose is stored in **mg/dL** everywhere in the database (`GlucoseReading.valueMgdl`, `Profile.targetLowMgdl`/`targetHighMgdl`) — see the design-notes comment at the top of `prisma/schema.prisma`. The user's display preference (`Profile.glucoseUnit: MGDL | MMOLL`) only affects presentation:

- Entry: `toMgdl(value, unit)` converts whatever the user typed (or an importer parsed) into mg/dL before it is ever written (`lib/domain/units.ts`).
- Display: `fromMgdl(mgdl, unit)` / `formatGlucoseWithUnit()` convert back at render time.
- The conversion factor (`MGDL_PER_MMOLL = 18.0182`) is defined once and used everywhere, so switching a user's unit preference is a pure display change with no data migration.
- Import connectors that don't declare a unit fall back to `inferGlucoseUnit()`, a heuristic based on the fact that real mmol/L values essentially never exceed 40 and real mg/dL values essentially never fall below 20.
- The analytics engine itself always computes in mg/dL, but statements shown to a user are worded in their preferred unit: `AnalyticsInput.displayUnit` (`lib/analytics/types.ts`) is threaded through from `Profile.glucoseUnit` (`lib/services/analytics-service.ts`), and every `Finding.statement`/`detail` string is built through `lib/analytics/format.ts`'s `formatLevel()`/`formatDelta()`, which converts only at the point of turning a number into words.

Every other quantity has one canonical storage unit too (mass in kg, volume in mL, duration in minutes — see the schema header comment), for the same reason.

## Timezone handling

All storage is UTC (every `DateTime` column). A user's `Profile.timezone` (IANA name, e.g. `America/Toronto`) is the only place "what day/hour was this in the user's own life" is computed, and it is always computed from UTC + the stored zone at read/write time — never stored as a local wall-clock string. `lib/domain/time.ts` provides the primitives:

- `zonedDateToUtc()` — interprets a `<input type="datetime-local">` string (a wall-clock value with no offset) in the user's zone and returns the correct UTC instant, via `timeZoneOffsetMs()`, which asks `Intl.DateTimeFormat` what the offset actually was at that instant (handles DST correctly, unlike a fixed offset).
- `dayKeyInZone()` / `hourInZone()` / `weekdayInZone()` — used throughout `lib/analytics/*` to bucket readings into the user's local days/hours (e.g. "post-dinner window", "weekday vs weekend") rather than UTC days, which would silently misclassify readings near midnight for most users.
- `toLocalInputValue()` — the inverse, so a form's default value shows the user's wall clock, not the server's.

## Where each concern lives

| Concern                        | Location                                                                       |
| ------------------------------ | ------------------------------------------------------------------------------ |
| Session issuance/verification  | `lib/auth/session.ts`                                                          |
| Password hashing/policy        | `lib/auth/password.ts`                                                         |
| Rate limiting                  | `lib/auth/rate-limit.ts`                                                       |
| Security audit log             | `lib/auth/audit.ts`                                                            |
| Per-user data access           | `lib/db/health-records.ts`, `lib/db/prisma.ts`                                 |
| Unit conversion/formatting     | `lib/domain/units.ts`                                                          |
| Clinical band classification   | `lib/domain/thresholds.ts`                                                     |
| Evidence grading thresholds    | `lib/domain/evidence.ts`                                                       |
| Dedupe key computation         | `lib/domain/dedupe.ts`                                                         |
| Timezone math                  | `lib/domain/time.ts`                                                           |
| Statistical analysis           | `lib/analytics/engine.ts`, `glucose.ts`, `associations.ts`, `stats.ts`, `ml/*` |
| Insight card assembly          | `lib/analytics/insights.ts`                                                    |
| AI provider abstraction        | `lib/ai/provider.ts`, `lib/ai/providers/*`                                     |
| AI orchestration               | `lib/ai/pipeline.ts`                                                           |
| AI safety enforcement          | `lib/ai/guardrails.ts`                                                         |
| AI structured output contracts | `lib/ai/schemas.ts`                                                            |
| AI data minimisation           | `lib/ai/redact.ts`                                                             |
| File parsing                   | `lib/import/parse.ts`                                                          |
| Per-vendor import connectors   | `lib/import/connectors/*`                                                      |
| Import dedupe                  | `lib/import/dedupe.ts`                                                         |
| App↔analytics/AI/import seam  | `lib/services/*`                                                               |
| Server Actions (mutations)     | `lib/actions/*`                                                                |
| Routes/pages                   | `app/**`                                                                       |
| Charts                         | `components/charts/*`                                                          |
| Security headers/CSP           | `next.config.ts`                                                               |
| Edge auth guard                | `middleware.ts`                                                                |
