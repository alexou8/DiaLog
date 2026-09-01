# DiaLog — Product Requirements Document

This document describes the product as it exists in the codebase today. Every
claim below is backed by a route, action, or module cited inline. Statements
about what is _not_ built are as load-bearing as statements about what is —
this is a health app, and overclaiming a capability here is worse than
omitting one. Where a claim is inferred rather than directly verified in code,
it is marked "(inferred)".

## 1. Purpose

DiaLog is a personal glucose and metabolic health tracker. It lets a person
log glucose readings and related health events by hand or import them from a
device/vendor export file, then surfaces statistically-graded patterns in
their own data, with an optional AI assistant that explains those patterns in
plain language. It is a self-tracking and pattern-surfacing tool, not a
diagnostic or treatment system.

## 2. Target users and core problems

- **People managing a metabolic condition day to day** (diabetes, prediabetes,
  or general glucose-conscious tracking) who already own a meter, CGM, or
  fitness/health export and want their own data in one place instead of
  scattered across a meter's app, a fitness app, and a paper log.
- **Users who want to see correlations in their own data** — e.g. how meals,
  sleep, or activity relate to glucose — without waiting for a clinician visit
  to notice a pattern, and without a tool overclaiming a pattern the data
  doesn't support.
- Problems addressed: fragmented data across devices/vendors (`lib/import/`),
  manual logging friction (nine record-type forms under `app/app/`, plus a
  combined quick-log flow at `app/app/quick-log/`), and the risk of
  over-interpreting small samples (`lib/domain/evidence.ts` grades every
  finding by sample size before it is shown).

## 3. Product boundaries and non-goals

**DiaLog is not a medical device.** This is the load-bearing constraint on
the rest of the product:

- It does not diagnose any condition.
- It does not recommend or adjust medication doses. This is enforced in code,
  not just policy: `lib/ai/guardrails.ts` runs intentionally over-broad regex
  filters against every AI response before display, and a match falls back to
  a safe template rather than reaching the user. A false rejection is
  preferred to a false negative here (a dosing instruction reaching a
  patient), so the filters are not "fixed" for apparent over-matching.
- It does not replace a healthcare provider, and states this in its own UI
  copy (`app/(marketing)/about/page.tsx`, onboarding disclaimer).

Explicit non-goals, verified against the codebase (absence of any
implementing route, action, or dependency):

- No live device sync — no OAuth or Bluetooth connection to a meter, CGM, or
  vendor cloud account. Every device integration is a **file import**
  initiated by the user (`lib/import/connectors/`). See
  `docs/DEVICE_INTEGRATIONS.md`.
- No push notifications, reminders, or background jobs — no queue, cron, or
  webhook infrastructure exists in the repo; analytics runs synchronously in
  the request path (`lib/services/analytics-service.ts`, called directly from
  `app/app/page.tsx` and `app/app/insights/page.tsx`).
- No care-team sharing, multi-user accounts, or clinician-facing views — the
  schema has no sharing, invite, or role model (`prisma/schema.prisma`
  contains only a single-owner `User` → records relationship).
- No account-recovery / password-reset flow (see §8 — a real gap, not a
  deliberate non-goal).

## 4. Major workflows

### Onboarding

`app/app/onboarding/page.tsx` + `app/app/onboarding/OnboardingForm.tsx` →
`lib/actions/onboarding.ts`, validated by `onboardingSchema`
(`lib/validation.ts`). Collects unit preference and baseline profile fields
and creates the user's `Profile` row. A new sign-up is routed here before the
rest of `app/app/**` is reachable (verified by the onboarding gate in
`app/app/layout.tsx` — not re-read in full for this document, but referenced
by `lib/actions/onboarding.ts`'s comments).

### Logging a reading

Nine record types each have a dedicated form and server action: glucose
(`app/app/glucose/new/`, `addGlucoseAction`), meal, exercise, sleep,
medication, weight, blood pressure, hydration, and mood/symptom
(`app/app/health/*/new/`, `app/app/meals/`, `app/app/activity/`), each backed
by its own Zod schema in `lib/validation.ts` and its own action in
`lib/actions/records.ts`. `app/app/quick-log/` offers a single freeform-text
entry point that the AI layer parses into a structured record proposal
(`proposeQuickLogAction`, `lib/actions/assistant.ts`) before the user confirms
it — the model never writes directly.

### Importing a file

`app/app/import/page.tsx` + `ImportPanel.tsx` → `lib/actions/import.ts`. Two
server actions implement a two-stage flow:

1. `analyzeImportAction` — parses an uploaded file through
   `lib/import/connectors/registry.ts`'s `detectConnector`, which tries every
   registered connector and falls back to a generic CSV/JSON/XML parser, then
   normalizes and dedupes in-memory and returns a preview (record counts,
   date range, row issues) without writing anything.
2. `commitImportAction` — writes the previously-analyzed batch, using
   content-addressed `dedupeKey`s (`lib/domain/dedupe.ts`,
   `lib/import/dedupe.ts`) so re-importing the same export file is a no-op.

`undoImportAction` reverses a committed `ImportBatch` by id, scoped to the
signed-in user.

### Reading insights and reports

`app/app/page.tsx` (dashboard) and `app/app/insights/page.tsx` both call
`lib/services/analytics-service.ts`, which runs `lib/analytics/engine.ts`
synchronously in the request path over the signed-in user's own records and
renders `InsightCard`s (`lib/analytics/insights.ts`) gated by evidence grade.
`app/app/reports/page.tsx` presents the same evidence-graded analytics as
charts with accompanying `<table>` data alternatives
(`components/charts/*`).

### Asking the assistant

`app/app/assistant/page.tsx` + `AssistantPanel.tsx` → `askAssistantAction`
(`lib/actions/assistant.ts`). The prompt is built from the user's
evidence-graded `AnalyticsResult`/`EvidenceBundle`, never from raw health
records (see `docs/AI_ARCHITECTURE.md`). Provider resolution is explicit arg
→ `AI_PROVIDER` env → `local`, and falls back to the local deterministic
provider if the configured one is unavailable. Every response passes Zod
schema validation, the guardrails regex filter, and a grounding check before
display.

### Exporting data

`app/api/export/route.ts` + `lib/services/export-service.ts`, reachable from
`app/app/settings/DataExport.tsx`. Two formats: JSON (all record types,
versioned schema) and per-type RFC4180 CSV. The route derives the user
strictly from the session — there is no `userId` query parameter — and every
export is written to the audit log (`audit({ userId: user.id, action:
'data.export', ... })`).

### Account and data deletion

`lib/actions/preferences.ts` exposes `deleteAccountAction` (deletes the user
and cascades) and a records-only deletion path that keeps the account and
settings (confirmed by its own success message, "Your account and settings
were kept."). Both are rate-limited (`guard(user.id, 'deleteaccount')`) and
logged to `AuditEvent` before the row that would identify the actor is gone.

## 5. Feature set

| Feature                                                                                                      | Status                                                                                                                                                              | Evidence                                                                                                                                                             |
| ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Manual logging: glucose, meal, exercise, sleep, medication, weight, blood pressure, hydration, mood, symptom | Implemented                                                                                                                                                         | `lib/actions/records.ts` (11 `add*Action` exports), one Zod schema per type in `lib/validation.ts`                                                                   |
| Quick-log (freeform text → structured proposal)                                                              | Implemented                                                                                                                                                         | `lib/actions/assistant.ts:proposeQuickLogAction`, `app/app/quick-log/`                                                                                               |
| Unit-aware glucose display (mg/dL ↔ mmol/L)                                                                 | Implemented                                                                                                                                                         | `lib/domain/units.ts` (`fromMgdl`/`toMgdl`, plausibility bounds), storage always mg/dL                                                                               |
| Target-range classification, non-colour-only                                                                 | Implemented                                                                                                                                                         | `lib/domain/thresholds.ts` (five bands, verified to pair icon/label with colour per README; not independently re-verified line-by-line here)                         |
| File import: LibreView, Nightscout, Apple Health, Omron, DiaLog legacy CSV, generic CSV/JSON/XML             | Implemented                                                                                                                                                         | `lib/import/connectors/registry.ts` `CONNECTORS` array (9 connectors)                                                                                                |
| Two-stage import (preview → commit)                                                                          | Implemented                                                                                                                                                         | `lib/actions/import.ts:analyzeImportAction` / `commitImportAction`                                                                                                   |
| Import dedupe (re-import is a no-op)                                                                         | Implemented                                                                                                                                                         | `lib/domain/dedupe.ts`, `lib/import/dedupe.ts`, `ImportBatch`/`ImportIssue` models                                                                                   |
| Import undo                                                                                                  | Implemented                                                                                                                                                         | `lib/actions/import.ts:undoImportAction`                                                                                                                             |
| Live device sync (Bluetooth / vendor OAuth)                                                                  | Not built                                                                                                                                                           | No connector performs network I/O; all take a parsed file (`lib/import/types.ts`)                                                                                    |
| Evidence-graded analytics (stats, associations, anomalies, trend, clustering, feature importance)            | Implemented                                                                                                                                                         | `lib/analytics/engine.ts` orchestrating `glucose`, `associations`, `stats`, `ml/*`; grading in `lib/domain/evidence.ts`                                              |
| Insights dashboard / reports with charts + table alternative                                                 | Implemented                                                                                                                                                         | `app/app/insights/page.tsx`, `app/app/reports/page.tsx`, `components/charts/*`                                                                                       |
| `Insight` DB model (persisted insight rows)                                                                  | Not built                                                                                                                                                           | `Insight` model exists in `prisma/schema.prisma:539` with zero read/write call sites in `lib/`, `app/` — `InsightCard`s are computed per-request and never persisted |
| AI assistant Q&A                                                                                             | Implemented                                                                                                                                                         | `lib/actions/assistant.ts:askAssistantAction`, local/Anthropic/OpenAI providers under `lib/ai/providers/`                                                            |
| AI weekly narrative                                                                                          | Implemented (per README; not independently traced end-to-end here — module presence in `lib/ai/prompts` and provider tests referenced by `docs/AI_ARCHITECTURE.md`) | `docs/AI_ARCHITECTURE.md`                                                                                                                                            |
| AI medical-safety guardrails (no dosing instructions)                                                        | Implemented                                                                                                                                                         | `lib/ai/guardrails.ts` regex filters, applied to every provider response                                                                                             |
| Zero-API-key local AI provider                                                                               | Implemented                                                                                                                                                         | `lib/ai/providers/local.ts`, deterministic, no network call                                                                                                          |
| Export: JSON (all types)                                                                                     | Implemented                                                                                                                                                         | `app/api/export/route.ts`, `lib/services/export-service.ts`                                                                                                          |
| Export: per-type RFC4180 CSV                                                                                 | Implemented                                                                                                                                                         | same files                                                                                                                                                           |
| Own email/password auth (bcrypt)                                                                             | Implemented                                                                                                                                                         | `lib/actions/auth.ts`, `lib/auth/password.ts`                                                                                                                        |
| Google OAuth sign-in                                                                                         | Implemented                                                                                                                                                         | `app/api/auth/google/{start,callback,disconnect}/route.ts`                                                                                                           |
| Session revocation ("sign out everywhere")                                                                   | Implemented                                                                                                                                                         | `User.tokenVersion`, `app/api/auth/sessions/revoke/route.ts`                                                                                                         |
| Password reset / "forgot password"                                                                           | **Not built**                                                                                                                                                       | `PasswordResetToken` model exists in `prisma/schema.prisma:85` but no route, action, or email-sending code references it anywhere in `app/` or `lib/`                |
| Rate limiting on auth/import/AI actions                                                                      | Implemented                                                                                                                                                         | `lib/auth/rate-limit.ts`, `RATE_LIMITS`, called from `lib/actions/auth.ts`, `lib/actions/import.ts`, `lib/actions/assistant.ts`, `lib/actions/preferences.ts`        |
| Audit log                                                                                                    | Implemented                                                                                                                                                         | `AuditEvent` model, `lib/auth/audit.ts`, called from export, account deletion, and auth flows                                                                        |
| Data deletion (records-only, and full account)                                                               | Implemented                                                                                                                                                         | `lib/actions/preferences.ts`                                                                                                                                         |
| Accessible hand-built SVG charts with `<table>` alternative                                                  | Implemented                                                                                                                                                         | `components/charts/*`, cross-checked against `docs/ACCESSIBILITY.md`                                                                                                 |
| Bilingual UI chrome (English/French)                                                                         | Partially implemented                                                                                                                                               | `lib/i18n/dictionaries.ts` covers shared nav chrome only; page bodies are English-only                                                                               |
| Installable PWA / offline app shell                                                                          | Implemented                                                                                                                                                         | `public/sw.js`, `app/offline/page.tsx` — caches only the public shell, never `/app` pages or API responses                                                           |
| Push notifications / reminders                                                                               | Not built                                                                                                                                                           | No notification API usage, no scheduling infrastructure found                                                                                                        |
| Care-team sharing / clinician views                                                                          | Not built                                                                                                                                                           | No sharing/invite/role model in `prisma/schema.prisma`                                                                                                               |
| Multi-user accounts (one login, multiple people)                                                             | Not built                                                                                                                                                           | `User` → records is a single-owner relation throughout the schema                                                                                                    |
| `ml/` Python pipeline in production                                                                          | Not built (research-only)                                                                                                                                           | Standalone Python project under `ml/`, not imported by any Node code, not part of `npm run build`                                                                    |

## 6. Functional requirements

Derived from what the code actually enforces, not aspirational rules.

- **Glucose values must be plausible before storage.** `isPlausibleGlucose`
  in `lib/domain/units.ts` bounds entries per display unit; `GLUCOSE_ENTRY_BOUNDS`
  defines the range. Storage unit is always mg/dL — display-unit conversion
  happens only through `lib/domain/units.ts`, never inline in a component
  (repository-wide invariant, checked by CLAUDE.md and consistent with the
  single `units.ts` import graph observed).
- **Every mutation is validated server-side.** Each `add*Action` in
  `lib/actions/records.ts` parses its `FormData` through a matching Zod
  schema in `lib/validation.ts` before any Prisma write; there is no
  client-only validation path.
- **Import never silently drops a row.** Every connector produces either a
  `NormalizedRecord` or a `RowIssue` for each input row (contract in
  `lib/import/types.ts`, enforced by connector tests referenced in
  `docs/DEVICE_INTEGRATIONS.md`); nothing is dropped without a surfaced
  reason.
- **Import commit is idempotent.** `dedupeKey` uniqueness
  (`lib/domain/dedupe.ts`) makes committing the same file twice a no-op
  rather than a duplicate set of records.
- **Analytics findings must state their evidentiary weight.** Every
  `Finding` produced by `lib/analytics/engine.ts` is graded
  INSUFFICIENT/EARLY/EMERGING/CONSISTENT by sample size
  (`lib/domain/evidence.ts`) before `lib/analytics/insights.ts` turns it into
  a displayable card; low-confidence findings render as "not enough data
  yet" rather than a claim.
- **Statistics primitives never return `NaN`/`Infinity`.** `stats.ts`
  (`lib/analytics/`) returns `null` for undefined results, per CLAUDE.md and
  consistent with the evidence-grading contract (not independently
  re-derived from source here).
- **The AI layer never receives raw health records.** Prompts are built
  exclusively from the evidence-graded `AnalyticsResult`/`EvidenceBundle`;
  free-text user notes are redacted (`lib/ai/redact.ts`) before reaching any
  `isExternal: true` provider without consent.
- **AI responses cannot contain dosing instructions.** Every response is
  checked against `lib/ai/guardrails.ts`'s regex filters and a grounding
  check before the user sees it; a match substitutes a safe template.
- **All data access is scoped to the authenticated user.** Export
  (`app/api/export/route.ts`) and every server action derive the user from
  the session (`requireUser()`), never from a request-supplied id.
- **Session validity is authoritative server-side, not at the edge.**
  `middleware.ts` verifies JWT signature and expiry only (no DB access at the
  edge); `requireUser()` is what enforces `tokenVersion` against the DB,
  which is what makes "sign out everywhere" actually revoke a cookie.

## 7. Non-functional requirements

**Accessibility.** Target is WCAG 2.2, enforced by an axe-core Playwright
suite (`tests/e2e/`, `@axe-core/playwright`) — see `docs/ACCESSIBILITY.md`.
Charts are hand-built inline SVG, each shipping a real `<table>` data
alternative and never encoding meaning in colour alone
(`components/charts/*`); the same colour-plus-icon-plus-label pattern is used
for glucose range bands (`lib/domain/thresholds.ts`). This is treated as part
of the definition of done for UI changes, not an optional pass.

**Privacy and security.** Passwords are bcrypt-hashed
(`lib/auth/password.ts`); sessions are signed JWT cookies
(`lib/auth/session.ts`, `jose`), requiring `AUTH_SECRET` ≥ 32 chars and
failing closed otherwise. Health data never leaves the app for the AI layer
unless it has been evidence-graded/redacted first (§6). Exports and account
deletion are scoped strictly to the authenticated user and written to
`AuditEvent`. See `docs/SECURITY.md`.

**Performance.** Analytics runs synchronously within the request that renders
the dashboard/insights/reports pages — there is no background job queue, so
response time scales with the user's own record count computed on each load
(`lib/services/analytics-service.ts`). No caching layer for analytics results
was found in the code (absence checked, not exhaustively).

**Rate limiting.** Auth, import, AI, and account-deletion actions are
rate-limited via `lib/auth/rate-limit.ts`. This limiter is in-memory and
scoped to a single server instance — it does not coordinate across multiple
instances/regions (stated directly in code comments and consistent with
README's "No multi-region shared rate limiting").

**Browser / PWA support.** Ships an installable PWA shell
(`public/manifest.json` (inferred from `public/sw.js` referencing manifest
behavior — not independently opened), `public/sw.js`) that caches only the
public marketing shell — never `/app` pages, never API responses, never
health data — so the app opens instantly offline to a dedicated offline page
(`app/offline/page.tsx`). No native mobile app; this is a responsive web app.

## 8. Known limitations

- **No password-reset / account-recovery flow.** A `PasswordResetToken`
  Prisma model exists (`prisma/schema.prisma:85`) but nothing in `app/` or
  `lib/` creates, emails, verifies, or consumes such a token. A user who
  forgets their password today has no self-service recovery path. This is a
  genuine gap, not a documented non-goal.
- **`Insight` schema model is unused.** `prisma/schema.prisma:539` defines an
  `Insight` table with zero application read or write sites; all insight
  cards are computed per-request in memory (`lib/analytics/insights.ts`) and
  never persisted under this model. Dead schema surface, not a bug, but worth
  tracking before it's mistaken for the persistence layer.
- **No push notifications, reminders, or background jobs** — analytics and
  all other work runs synchronously in the request path.
- **No care-team sharing or clinician-facing views.**
- **No multi-user accounts** — one login maps to exactly one person's data.
- **French translation is chrome-only** — page bodies are English-only; the
  app states this rather than silently mixing languages.
- **Rate limiter is per-instance, in-memory** — does not coordinate across
  multiple server processes or regions, so its effective limit is looser than
  configured under horizontal scaling.
- **No live device sync of any kind** — every device/vendor path is a
  user-initiated file import; see `docs/DEVICE_INTEGRATIONS.md` for which
  vendors have genuine live APIs (e.g. Dexcom, Nightscout) that are
  deliberately not implemented here.
- **`ml/` is a standalone research pipeline** — not deployed, does not run as
  part of the product, and has no code path connecting it to `app/` or
  `lib/`.

## 9. Future opportunities

None of the items below are built, scheduled, or committed to. They are
directions the current architecture does not preclude, not a roadmap.

- A real password-reset flow using the existing (currently unused)
  `PasswordResetToken` model as its foundation.
- Persisting computed insights via the existing (currently unused) `Insight`
  model, enabling history/trend-of-insights views instead of recomputing on
  every page load.
- Background/scheduled analytics computation, to decouple page-load latency
  from record count.
- Push notifications or reminder scheduling for logging cadence.
- Live device sync for vendors with genuine public APIs (Dexcom, Nightscout)
  where only file import exists today.
- Care-team sharing or a read-only clinician view.
- Full-body (not just chrome) French localization, or additional languages.
- Distributed/shared rate limiting for multi-instance deployments.
- Promoting relevant pieces of the `ml/` research pipeline into the
  production analytics path, if and when their outputs can be evidence-graded
  to the same standard as `lib/analytics/`.
