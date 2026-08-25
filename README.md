# DiaLog

DiaLog is a personal glucose and metabolic health tracker: a Next.js web app for logging glucose readings, meals, activity, sleep, medications and other health events, importing them from device/vendor exports, and surfacing statistically-graded patterns in your own data — with an optional AI assistant that explains those patterns in plain language.

**DiaLog is not a medical device.** It does not diagnose any condition, does not recommend or adjust medication doses, and does not replace a healthcare provider. Every pattern it surfaces is graded by how much of your own data supports it, and it says "not enough data yet" when that's the honest answer. See [Limitations and medical disclaimer](#limitations-and-medical-disclaimer) below.

## Screenshots

|                                                               |                                                                  |
| ------------------------------------------------------------- | ---------------------------------------------------------------- |
| ![Landing page](docs/screenshots/landing.png)                 | ![Dashboard](docs/screenshots/dashboard.png)                     |
| ![Dashboard, dark theme](docs/screenshots/dashboard-dark.png) | ![Glucose log](docs/screenshots/glucose.png)                     |
| ![Add a reading](docs/screenshots/add-reading.png)            | ![Insights](docs/screenshots/insights.png)                       |
| ![Reports](docs/screenshots/reports.png)                      | ![History](docs/screenshots/history.png)                         |
| ![Data import](docs/screenshots/import.png)                   | ![AI assistant](docs/screenshots/assistant.png)                  |
| ![Settings](docs/screenshots/settings.png)                    | ![Mobile dashboard](docs/screenshots/mobile-dashboard.png)       |
| ![Mobile glucose log](docs/screenshots/mobile-glucose.png)    | ![Mobile add a reading](docs/screenshots/mobile-add-reading.png) |

## Features

Only features that are actually implemented and working are listed here.

- **Glucose, meal, exercise, sleep, medication, weight, blood pressure, hydration, symptom and mood logging** — manual entry forms for every record type in `prisma/schema.prisma`, each with server-side validation (`lib/validation.ts`).
- **Unit-aware glucose display** — canonical storage is mg/dL; the UI converts to mmol/L per user preference (`lib/domain/units.ts`), with plausibility bounds on entry.
- **Target-range classification with plain-language, non-colour-only bands** — `lib/domain/thresholds.ts` classifies each reading into five bands (very low → very high) with an icon and label in addition to colour.
- **Evidence-graded analytics** — `lib/analytics/engine.ts` runs summary statistics, meal/activity/sleep associations, anomaly detection, trend detection, day-pattern clustering and feature-importance estimation over a user's own data, and every finding is graded INSUFFICIENT/EARLY/EMERGING/CONSISTENT by sample size (`lib/domain/evidence.ts`) before it is shown anywhere.
- **AI assistant (question answering, weekly narrative, quick-log parsing)** — works with **zero API keys** via a deterministic, no-network local provider (`lib/ai/providers/local.ts`); optionally uses Anthropic or OpenAI if configured. Every response passes schema validation, a medical-safety regex filter, and a grounding check before display (`lib/ai/guardrails.ts`).
- **File-based data import** — CSV, XLSX, JSON and XML, with dedicated connectors for the Abbott LibreView export, Nightscout `entries.json`, Apple Health `export.xml`, Omron CSV exports, and DiaLog's own legacy CSV, plus generic CSV/JSON/XML fallbacks (`lib/import/connectors/*`). Two-stage flow: parse-and-preview, then commit — nothing is written until the user confirms. See [docs/DEVICE_INTEGRATIONS.md](docs/DEVICE_INTEGRATIONS.md) for exactly what is and isn't real per vendor.
- **Content-addressed dedupe on import** — re-importing the same export is a no-op rather than creating duplicates (`lib/domain/dedupe.ts`, `lib/import/dedupe.ts`).
- **Full data export** — JSON (all record types, versioned schema) and per-type RFC4180 CSV, scoped strictly to the signed-in user (`app/api/export/route.ts`, `lib/services/export-service.ts`).
- **Own signed-cookie authentication** — email/password with bcrypt hashing, session revocation via a `tokenVersion` counter, and per-key rate limiting on sign-in/sign-up/import/AI calls (`lib/auth/*`).
- **Hand-built, accessible SVG charts** — every chart ships a real `<table>` data alternative and never encodes meaning in colour alone (`components/charts/*`). See [docs/ACCESSIBILITY.md](docs/ACCESSIBILITY.md).
- **Bilingual chrome** — English and French (Canada) UI chrome via `lib/i18n/dictionaries.ts`; page bodies are English-only today, which the app states rather than hiding.
- **Installable PWA shell** — a manifest and a service worker that caches only the public app shell (never `/app` pages or API responses, never health data) so the app opens instantly offline to a real offline page (`public/sw.js`).
- **Seeded demo account** — `npm run db:seed` creates a synthetic three-month history for exercising every screen without real data.

## What is not built

- **No push notifications, reminders, or background jobs.** Analytics runs synchronously in the request path (see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)).
- **No live device sync.** Every integration is a file import initiated by the user; there is no OAuth or Bluetooth connection to a meter, CGM, or vendor cloud account. See [docs/DEVICE_INTEGRATIONS.md](docs/DEVICE_INTEGRATIONS.md) for what is and isn't real per vendor (Dexcom and Nightscout have genuine live APIs that are deliberately not implemented here).
- **No multi-region shared rate limiting.** The rate limiter is in-memory and per server instance — see [docs/SECURITY.md](docs/SECURITY.md).
- **No care-team sharing, multi-user accounts, or clinician-facing views.**
- **No dose calculation or recommendation of any kind, ever**, by design — enforced by regex guardrails in `lib/ai/guardrails.ts`, not just policy.
- **French translation of page bodies** — only shared navigation chrome is translated; page content is English-only.
- **The `ml/` Python pipeline is not deployed and does not run in the product** — see below.

## Tech stack

| Layer          | Choice (pinned version)                                                     | Why                                                                                                                                                                                                                                   |
| -------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Framework      | Next.js **15.5.4** (App Router), React **19.1.1**                           | Server Components let pages fetch and render user data directly without a separate API layer for most reads; Server Actions (`lib/actions/*`) give validated, CSRF-safe mutations without hand-rolling a REST/GraphQL API.            |
| Database       | PostgreSQL via Prisma **6.16.2** (`@prisma/client`)                         | Health records are relational, per-user, and need real transactions (e.g. import commit, export) — Prisma gives typed queries and migrations against a real ACID database rather than a document store.                               |
| Auth           | Hand-rolled signed-cookie sessions (`jose` **6.1.0**, `bcryptjs` **3.0.2**) | No third-party identity provider sees health data or account existence; a stateless JWT cookie plus a `tokenVersion` column gives "sign out everywhere" without a session-store dependency. See [docs/SECURITY.md](docs/SECURITY.md). |
| Charts         | Hand-built inline SVG (`components/charts/*`), no charting library          | Full control over the accessibility contract: every chart ships a real `<table>` alternative and never relies on colour alone — properties a generic charting library's default output does not guarantee.                            |
| Styling        | Tailwind CSS **4.1.13** (`@tailwindcss/postcss`)                            | CSS-first configuration (no `tailwind.config.js`) and native cascade layers keep the design-token/dark-mode/reduced-motion logic in `app/globals.css` simple and inspectable.                                                         |
| Validation     | Zod **3.25.76**                                                             | Every form, server action and API input is parsed through one schema library (`lib/validation.ts`, `lib/ai/schemas.ts`), so validation logic is centralized and typed.                                                                |
| Import parsing | `exceljs` **4.4.0**, `fast-xml-parser` **5.2.5**                            | XLSX and XML are real vendor export formats (Apple Health, LibreView) that need dedicated parsers, not just a CSV reader.                                                                                                             |
| Testing        | Vitest **3.2.4**, Playwright **1.55.0**, `@axe-core/playwright` **4.10.2**  | Vitest for fast unit/integration tests against real logic; Playwright + axe-core for browser-level and automated accessibility checks.                                                                                                |

## Quick start

### Prerequisites

- Node.js ≥ 20 (see `engines` in `package.json`)
- A local PostgreSQL instance (any recent version; the schema uses no exotic features)

### 1. Install

```bash
npm install
```

`postinstall` runs `prisma generate` automatically.

### 2. Configure environment

```bash
cp .env.example .env
```

Generate `AUTH_SECRET` (must be at least 32 characters — `lib/auth/session.ts` throws at runtime otherwise):

```bash
node -e "console.log(require('crypto').randomBytes(48).toString('base64url'))"
```

Point `DATABASE_URL` (and `DIRECT_DATABASE_URL`) at your local Postgres. The default in `.env.example` assumes a `dialog` database on `127.0.0.1:5432` with user/password `postgres`/`dialog` — adjust to match your local setup, e.g.:

```bash
createdb dialog
```

### 3. Apply migrations

```bash
npx prisma migrate deploy
```

### 4. Seed demo data (optional but recommended)

```bash
npm run db:seed
```

This creates a synthetic three-month account:

- Email: `demo@dialog.health`
- Password: `demo-account-2026`

This data is entirely synthetic (generated by a seeded PRNG in `prisma/seed.ts`) and carries no clinical meaning.

### 5. Run

```bash
npm run dev
```

Visit `http://localhost:3000`.

## npm scripts

| Script                            | What it does                                                                                                       |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `npm run dev`                     | Starts the Next.js dev server.                                                                                     |
| `npm run build`                   | Runs `prisma generate` then `next build` (see [Deployment](#deployment-on-vercel) for why generate is chained in). |
| `npm start`                       | Serves the production build.                                                                                       |
| `npm run lint`                    | `next lint` (ESLint 9, flat config).                                                                               |
| `npm run typecheck`               | `tsc --noEmit` — strict mode, `noUncheckedIndexedAccess` on.                                                       |
| `npm run format` / `format:check` | Prettier write / check.                                                                                            |
| `npm test`                        | `vitest run` — the DB-free unit suite only (`tests/unit/**`). Works with no Postgres instance.                     |
| `npm run test:watch`              | Same suite, watch mode.                                                                                            |
| `npm run test:e2e`                | `playwright test` — browser end-to-end tests.                                                                      |
| `npm run db:migrate`              | `prisma migrate dev` — create/apply a migration in development.                                                    |
| `npm run db:deploy`               | `prisma migrate deploy` — apply pending migrations (production-safe, no shadow database).                          |
| `npm run db:push`                 | `prisma db push` — sync schema without a migration file (prototyping only).                                        |
| `npm run db:seed`                 | `tsx prisma/seed.ts` — populate the demo account.                                                                  |
| `npm run db:studio`               | `prisma studio` — browse the database.                                                                             |

There is no `npm run test:integration` script in `package.json`; the real-database integration/security suite (`tests/integration/**`) is run directly:

```bash
npx vitest run --config vitest.integration.config.ts
```

(see [Testing](#testing) below).

## Environment variables

Source of truth: `.env.example`, cross-checked against every `process.env.*` reference in the codebase.

| Variable                                         | Required?                   | Default                                                                 | What it does                                                                                                                                                                                                                                                                                                                              |
| ------------------------------------------------ | --------------------------- | ----------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DATABASE_URL`                                   | Yes                         | —                                                                       | PostgreSQL connection string Prisma reads/writes through (use a pooled URL on Vercel).                                                                                                                                                                                                                                                    |
| `DIRECT_DATABASE_URL`                            | Yes (for migrations)        | —                                                                       | Non-pooled connection Prisma's `directUrl` uses for `prisma migrate`; a pooler (PgBouncer/Vercel's pooled Postgres) cannot run migrations.                                                                                                                                                                                                |
| `AUTH_SECRET`                                    | Yes                         | —                                                                       | ≥32-byte secret used to sign session-cookie JWTs (`lib/auth/session.ts`); the app throws at first use if missing or too short.                                                                                                                                                                                                            |
| `AI_PROVIDER`                                    | No                          | `local`                                                                 | One of `anthropic` \| `openai` \| `local`. Selects the AI provider (`lib/ai/provider.ts`). `local` never leaves the server and needs no key.                                                                                                                                                                                              |
| `ANTHROPIC_API_KEY`                              | No                          | unset                                                                   | API key for the Anthropic provider (`lib/ai/providers/anthropic.ts`). Without it, `anthropic` is requested but unavailable and the app falls back to `local`.                                                                                                                                                                             |
| `ANTHROPIC_MODEL`                                | No                          | `claude-sonnet-5`                                                       | Model id sent to the Anthropic Messages API.                                                                                                                                                                                                                                                                                              |
| `OPENAI_API_KEY`                                 | No                          | unset                                                                   | API key for the OpenAI provider (`lib/ai/providers/openai.ts`). Same fallback behaviour as above.                                                                                                                                                                                                                                         |
| `OPENAI_MODEL`                                   | No                          | `gpt-4o-mini`                                                           | Model id sent to the OpenAI API.                                                                                                                                                                                                                                                                                                          |
| `NEXT_PUBLIC_APP_URL`                            | No                          | `http://localhost:3000`                                                 | Used as `metadataBase` for absolute URLs in page metadata (`app/layout.tsx`).                                                                                                                                                                                                                                                             |
| `NODE_ENV`                                       | Set by the runtime          | —                                                                       | Also read directly: enables `'unsafe-eval'` in the dev CSP (`next.config.ts`), toggles the `secure` cookie flag (`lib/auth/session.ts`), toggles Prisma query logging (`lib/db/prisma.ts`), and gates service-worker registration to production (`components/ServiceWorkerRegistration.tsx`). You do not set this yourself in normal use. |
| `TEST_DATABASE_URL` / `TEST_DIRECT_DATABASE_URL` | No (integration tests only) | `postgresql://postgres:dialog@127.0.0.1:5432/dialog_test?schema=public` | Points the integration/security Vitest suite at an isolated `dialog_test` database (`tests/integration/setup-env.ts`); the suite refuses to run against anything whose name doesn't match `dialog_test`.                                                                                                                                  |

## Testing

| Layer       | Command                                                                         | What it actually covers                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ----------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Unit        | `npm test` (`vitest run`, `tests/unit/**`, 29 test files)                       | Pure logic with no database or network: domain rules (units, thresholds, evidence grading, dedupe keys), the full analytics engine (stats, associations, ML sub-modules), every AI guardrail and schema, and every import connector against real fixture files in `tests/fixtures/import/`. No Postgres instance required.                                                                                                                                          |
| Integration | `npx vitest run --config vitest.integration.config.ts` (`tests/integration/**`) | Round-trips every `RECORD_TYPES` kind against a real, isolated `dialog_test` Postgres database, exercising `lib/db/health-records.ts` (creation, retrieval, deletion, per-user scoping). Runs strictly sequentially (`fileParallelism: false`, single fork) because test files share one physical database. `tests/integration/setup-env.ts` hard-refuses to run against anything not named `dialog_test`, so it cannot touch a real dev/prod database by accident. |
| E2E         | `npm run test:e2e` (`playwright test`, `tests/e2e/**`, 7 spec files)            | Real browser flows against a built app on port 3100: auth/onboarding, logging, dashboard/insights, import, the assistant, a keyboard-only task completion, mobile viewports, and automated accessibility checks (`@axe-core/playwright` against WCAG 2.2 AA tags) on every public and authenticated page — see [docs/ACCESSIBILITY.md](docs/ACCESSIBILITY.md).                                                                                                      |

Nothing in this repository runs a full test suite against production data, and the unit suite (the one CI would run without provisioning a database) never touches Postgres at all.

## Deployment on Vercel

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Falexou8%2FDiaLog&env=DATABASE_URL,DIRECT_DATABASE_URL,AUTH_SECRET&envDescription=DATABASE_URL%20and%20DIRECT_DATABASE_URL%20are%20PostgreSQL%20connection%20strings%3B%20AUTH_SECRET%20is%20a%2032%2B%20character%20random%20value%20used%20to%20sign%20session%20cookies&project-name=dialog&repository-name=dialog)

DiaLog needs a PostgreSQL database and one secret before it will run anywhere. Nothing else is required — the assistant defaults to the built-in local engine, so no AI keys are needed.

| Variable              | Required | What to put in it                                                                                                         |
| --------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------- |
| `DATABASE_URL`        | Yes      | Pooled PostgreSQL connection string.                                                                                      |
| `DIRECT_DATABASE_URL` | Yes      | Direct (non-pooled) connection string, used by `prisma migrate`. If your provider gives only one string, use it for both. |
| `AUTH_SECRET`         | Yes      | 32+ random characters. Generate with `node -e "console.log(require('crypto').randomBytes(48).toString('base64url'))"`.    |
| `AI_PROVIDER`         | No       | `local` (default), `anthropic` or `openai`.                                                                               |
| `NEXT_PUBLIC_APP_URL` | No       | Public URL of the deployment, used for canonical metadata.                                                                |

After the first deploy, apply the schema once against the direct URL:

```bash
DATABASE_URL="$DIRECT_DATABASE_URL" npx prisma migrate deploy
```

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for the full procedure, including where migrations belong in a deploy pipeline and why `prisma generate` runs during the build.

## Project structure

```
DiaLog/
├── app/                          # Next.js App Router routes
│   ├── (marketing)/              # Public pages: landing, about, privacy, security, accessibility, help, terms
│   ├── (auth)/                   # Sign-in / sign-up (redirect to /app if already authenticated — middleware.ts)
│   ├── app/                      # Authenticated app shell — every route here requires a session
│   │   ├── page.tsx              # Dashboard: recent readings, insight cards, quick stats
│   │   ├── glucose/               onboarding, quick-log, glucose, meals, activity, health/*,
│   │   ├── meals/                 history, insights, reports, import, assistant, settings
│   │   ├── activity/
│   │   ├── health/                # blood-pressure, medication, mood, sleep, weight sub-forms
│   │   ├── history/
│   │   ├── insights/
│   │   ├── reports/
│   │   ├── import/
│   │   ├── assistant/
│   │   ├── quick-log/
│   │   ├── onboarding/
│   │   └── settings/
│   └── api/                      # REST-style routes: export/route.ts (data export) and health/route.ts (liveness/readiness probe)
├── components/
│   ├── charts/                   # Hand-built accessible SVG charts (GlucoseTimeline, Sparkline, BarChart, RangeBar)
│   ├── ui/                       # Shared primitives (Card, Badge, Stat, form controls, WhyThis evidence panel)
│   └── *.tsx                     # AppNav, ThemeToggle, InsightCardView, GlucoseReadingRow, service worker registration
├── lib/
│   ├── actions/                  # 'use server' Server Actions: auth, records, import, preferences, onboarding, assistant
│   ├── domain/                   # Pure domain rules: units, thresholds, evidence grading, dedupe keys, timezone math
│   ├── analytics/                # Statistical engine: engine.ts orchestrates glucose stats, associations, and ml/*
│   │   └── ml/                   # anomaly, cluster, importance, trend — in-process statistical routines, not a served model
│   ├── ai/                       # Provider abstraction, prompts, guardrails, schemas, redaction, pipeline orchestration
│   │   └── providers/            # anthropic.ts, openai.ts, local.ts (no-network default)
│   ├── import/                   # File parsing + connectors + dedupe + summary
│   │   └── connectors/           # One file per vendor/format (see docs/DEVICE_INTEGRATIONS.md)
│   ├── services/                 # Application-layer seams: analytics-service, import-service, export-service
│   ├── db/                       # Prisma client singleton + scoped health-record queries
│   ├── auth/                     # Session (JWT cookie), password hashing/policy, rate limiting, audit log
│   └── i18n/                     # en-CA / fr-CA dictionaries
├── prisma/
│   ├── schema.prisma              # Full data model (see docs/ARCHITECTURE.md)
│   ├── migrations/
│   └── seed.ts                    # Synthetic demo-account generator
├── tests/
│   ├── unit/                      # DB-free unit tests (`npm test`)
│   ├── integration/                # Real-Postgres integration/security tests
│   └── fixtures/import/            # Real sample export files used by connector tests
├── ml/                             # Offline Python research pipeline — NOT deployed, NOT imported by the app (see docs/ML_PIPELINE.md)
├── docs/                           # This documentation set
├── middleware.ts                   # Edge session guard for /app and the auth pages
└── next.config.ts                  # Security headers + CSP
```

## Data import

DiaLog imports data from files the user exports themselves — there is no live device or account sync. Supported formats: CSV (RFC4180, with delimiter and BOM auto-detection), XLSX, JSON, and XML, routed through per-vendor connectors where a real layout is known (Abbott LibreView, Nightscout `entries.json`, Apple Health `export.xml`, Omron CSV, DiaLog's own legacy export) with generic CSV/JSON/XML connectors as a fallback (`lib/import/connectors/registry.ts`). Every import is a two-stage parse-then-preview-then-commit flow, and content-addressed dedupe keys make re-importing the same file a no-op. Full details on what is verified vs. unverified per vendor, and what integration paths (Dexcom, Nightscout live API) exist but are deliberately not built, are in **[docs/DEVICE_INTEGRATIONS.md](docs/DEVICE_INTEGRATIONS.md)**.

## AI architecture

The AI assistant only ever reasons over a small, pre-aggregated `EvidenceBundle` — never raw readings, meals, or free text — enforced at runtime by `lib/ai/pipeline.ts`'s `assertNoRawRecords`. It works fully offline via a deterministic local provider and optionally calls Anthropic or OpenAI if configured, with every response passing schema validation, a medical-safety filter, and a grounding check before being shown. Full details, the evidence-grading thresholds, and the "not enough data" behaviour are in **[docs/AI_ARCHITECTURE.md](docs/AI_ARCHITECTURE.md)**.

## Privacy and security

Sessions are signed, stateless cookies with server-side revocation via a `tokenVersion` counter; every query is scoped by the authenticated user's id; passwords are bcrypt-hashed; uploads are size-capped and parsed defensively; a CSP and standard security headers are set on every response; and health data is never sent to an external AI provider without explicit consent. See **[docs/SECURITY.md](docs/SECURITY.md)** for the full threat model, controls, and a "deploying this for real" checklist.

## Accessibility

DiaLog targets WCAG 2.2 AA: no colour-only meaning, real `<table>` alternatives for every chart, visible focus states, 44px minimum touch targets, respect for reduced-motion and OS/user text-scaling preferences, and semantic landmark structure. See **[docs/ACCESSIBILITY.md](docs/ACCESSIBILITY.md)** for the concrete implementation and known gaps.

## What is genuinely implemented vs. future opportunities

| Genuinely implemented today                                                                                 | Future opportunity (not built)                                                                                                                           |
| ----------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Manual logging for 10 record types with server-side validation                                              | Live device/CGM sync (Dexcom and Nightscout both have real APIs — see docs/DEVICE_INTEGRATIONS.md)                                                       |
| File import for CSV/XLSX/JSON/XML with 5 dedicated + 3 generic connectors                                   | Bluetooth/USB direct meter connection (researched and explicitly rejected as unreliable — see docs/DEVICE_INTEGRATIONS.md)                               |
| Evidence-graded statistical findings (summary, association, trend, anomaly, clustering, feature importance) | A served, continuously-retrained ML model (the `ml/` pipeline is offline research only, never deployed)                                                  |
| AI assistant with a working no-network local provider as the default                                        | Free-form clinical Q&A beyond the guardrailed, evidence-grounded scope (deliberately out of scope, not a gap)                                            |
| Anthropic/OpenAI providers, both implemented and unit-tested                                                | Additional providers (Gemini, local LLM runtime, etc.)                                                                                                   |
| Medical-safety guardrails (dosing/diagnosis regex filters, grounding check, claim consistency check)        | Human-reviewed clinical-safety audit of the guardrail pattern list                                                                                       |
| Full JSON + per-type CSV data export, scoped to the signed-in user                                          | Scheduled/automatic export or backup                                                                                                                     |
| Own signed-cookie auth with sign-out-everywhere revocation                                                  | Multi-factor authentication, OAuth/SSO sign-in, passkeys                                                                                                 |
| Per-key in-memory rate limiting on sign-in/up, import, AI, and export                                       | Shared/distributed rate limiting for multi-instance or multi-region deployment                                                                           |
| Hand-built accessible SVG charts with table alternatives                                                    | Third-party accessibility audit beyond automated axe-core checks                                                                                         |
| English + French (Canada) navigation chrome                                                                 | Full page-body translation; additional locales                                                                                                           |
| Installable PWA shell with an offline page                                                                  | Push notifications, background sync, reminders                                                                                                           |
| 90-day demo dataset via `db:seed`                                                                           | Any production seed/import of real historical data at account creation                                                                                   |
| Content-addressed import dedupe                                                                             | Cross-device conflict resolution (e.g. the same physical event logged manually and later imported still creates two records if their dedupe keys differ) |
| Security headers + CSP on every response                                                                    | Formal penetration test                                                                                                                                  |
| Audit log for security-relevant actions (never health values)                                               | User-facing security/activity log UI                                                                                                                     |

## Limitations and medical disclaimer

DiaLog is an informational tracking and pattern-analysis tool. It is **not** a medical device, does not diagnose any condition, and does not recommend, calculate, or adjust medication doses — this is enforced in code (`lib/ai/guardrails.ts`'s medical-safety filter blocks dosing/diagnostic language from the AI assistant, and `MedicationEvent` in the schema is tracking-only). Every statistical finding is graded by sample size against thresholds in `lib/domain/evidence.ts`, and findings below the minimum sample size are never surfaced as confident claims. Nothing in this app should be used to make a treatment decision; always consult a qualified healthcare provider. The `ml/` directory is an offline research pipeline whose synthetic-data results carry no clinical validity — see [docs/ML_PIPELINE.md](docs/ML_PIPELINE.md).

## Licence

MIT — see [LICENSE](LICENSE).
