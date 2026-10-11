# Deployment

## Local development

1. Install Node.js and a local PostgreSQL instance. Use Node 22: `package.json`
   `engines` allows `>=20`, but CI, `seed-demo.yml` and `.nvmrc` pin 22, so 22 is
   the only version that is tested. (Vercel resolves `>=20` to its newest
   supported major, which CI has not run.)
2. `npm ci` (runs `prisma generate` via `postinstall`).
3. `cp .env.example .env` and fill in `DATABASE_URL` / `DIRECT_DATABASE_URL` (point both at the same local database for development — there is no pooler to distinguish them from) and `AUTH_SECRET` (generate with `node -e "console.log(require('crypto').randomBytes(48).toString('base64url'))"`).
4. `npx prisma migrate deploy` to apply migrations (or `npm run db:migrate` — `prisma migrate dev` — if you intend to author a new migration).
5. `npm run db:seed` to populate the demo account (`demo@dialog.health` / `demo-account-2026`) with three months of synthetic data.
6. `npm run dev` and visit `http://localhost:3000`.

See the README's [Quick start](../README.md#quick-start) for the same steps with more explanation.

## Environment variables

The canonical list. `.env.example` carries the same names with comments; the
README and AGENTS.md summarise this table rather than restate it.

| Variable                                                                 | Status                                  | Notes                                                                                                                                                                                                                                                                           |
| ------------------------------------------------------------------------ | --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DATABASE_URL`                                                           | **Required**                            | Runtime connection. Pooled on Vercel.                                                                                                                                                                                                                                           |
| `DIRECT_DATABASE_URL`                                                    | **Required**                            | Non-pooled connection used by Prisma Migrate. `prisma/schema.prisma` declares `directUrl = env("DIRECT_DATABASE_URL")`, so every Prisma CLI command (generate, migrate, validate, studio) errors while it is unset. Use the same value as `DATABASE_URL` if you have no pooler. |
| `AUTH_SECRET`                                                            | **Required**                            | At least 32 characters; the app fails closed without it. A fresh value per environment.                                                                                                                                                                                         |
| `NEXT_PUBLIC_APP_URL`                                                    | **Required for password recovery**      | The trusted public origin reset links are built from; never derived from the request Host. HTTPS except loopback. Also the base of the Google redirect URI and the site's `metadataBase`.                                                                                       |
| `EMAIL_PROVIDER`, `EMAIL_HTTP_URL`, `EMAIL_HTTP_TOKEN`, `EMAIL_FROM`     | **Required in production for recovery** | See [Password recovery mail](#password-recovery-mail). The default `console` provider sends nothing in production, so recovery is silently unavailable.                                                                                                                         |
| `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`                               | Optional                                | Both set enables Google sign-in; both empty disables it.                                                                                                                                                                                                                        |
| `AI_PROVIDER`                                                            | Optional                                | `local` (default) \| `anthropic` \| `openai`.                                                                                                                                                                                                                                   |
| `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`, `OPENAI_API_KEY`, `OPENAI_MODEL` | Optional                                | Only read when that provider is selected.                                                                                                                                                                                                                                       |
| `VERCEL_URL`, `VERCEL_PROJECT_PRODUCTION_URL`                            | Set by Vercel                           | Fallback origin for `metadataBase` when `NEXT_PUBLIC_APP_URL` is empty (`app/layout.tsx`). Do not set by hand.                                                                                                                                                                  |
| `TEST_DATABASE_URL`, `TEST_DIRECT_DATABASE_URL`                          | Test only                               | Integration suite; the database must be named `dialog_test`.                                                                                                                                                                                                                    |
| `E2E_DATABASE_URL`, `E2E_DIRECT_DATABASE_URL`                            | Test only                               | E2E suite; the database must be named `dialog_e2e` (or `dialog_test`) because the suite deletes every account in it. Never `DATABASE_URL`.                                                                                                                                      |
| `PW_SEED`                                                                | Test only                               | Replays an e2e run's generated data; printed at the start of every run.                                                                                                                                                                                                         |

## Password recovery mail

Recovery requires a usable `AUTH_SECRET` and `NEXT_PUBLIC_APP_URL`, which is
the trusted public origin used in mailed links. The origin is never derived
from a request's Host header. Use HTTPS; HTTP is accepted only on loopback for
local development/testing.

Configure these settings before enabling production recovery:

| Variable              | Meaning                                                                             |
| --------------------- | ----------------------------------------------------------------------------------- |
| `EMAIL_PROVIDER`      | `http` for a real mail relay; default `console` is development/test only            |
| `EMAIL_HTTP_URL`      | Authenticated relay endpoint; HTTPS for remote hosts, HTTP allowed only on loopback |
| `EMAIL_HTTP_TOKEN`    | Secret sent as `Authorization: Bearer …` to the relay                               |
| `EMAIL_FROM`          | Sender address authorized by your relay                                             |
| `NEXT_PUBLIC_APP_URL` | Public app origin used to build reset links                                         |

**The HTTP transport has never been exercised against a real mail provider.**
The only thing that has ever received a message from it is the loopback receiver
the e2e suite runs on `127.0.0.1:3211`. Treat the first production send as
unverified, and confirm delivery with a dedicated non-health-data account before
relying on recovery.

The built-in HTTP transport is a concrete, vendor-independent relay client,
not a direct Resend/SendGrid/SMTP integration. Provision a relay that accepts
`POST` JSON `{ "from": "…", "to": "…", "subject": "…", "text": "…" }`
and sends that message through your chosen mail service. A 2xx response means
accepted for delivery; non-2xx, network errors, redirects, or the 10-second
deadline are failures. Configure relay authentication, sender/domain
verification, delivery monitoring and SPF/DKIM/DMARC at that service. No mail
dependency is added to the application. Never expose this relay anonymously.

`console` prints a reset URL to stdout **only** when `NODE_ENV ===
'development'`. Under `test`, it exposes a bounded in-memory capture through
`takeCapturedEmails()`; it never prints. Under production it reports itself
unavailable and throws a typed failure if invoked. Unknown providers and
missing configuration fail closed before account lookup, and the request form
shows recovery unavailable. A runtime delivery failure keeps the public
confirmation neutral to avoid account enumeration, invalidates its token, and
emits a fixed operator error plus a failure audit. Monitor those signals:
delivery uses Next 15's `after()` lifecycle, not a detached promise. The host
must support Next's `waitUntil` lifecycle and allow sufficient function duration
for the transaction and the relay's 10-second deadline. There is no durable
outbox or automatic retry; a process crash or host deadline can still interrupt
work even though the response has already been sent.

Trust the mail transport as an account-recovery credential processor. Disable
click tracking and mail-body logging. Exclude query strings for
`/api/auth/password/reset` and `/reset-password` from platform/proxy/APM logs;
both emailed tokens and signed form proofs are bearer credentials. Do not
record recovery pages in analytics. Mailbox control is sufficient to recover
a password account, and existing email addresses have never been verified;
see [Security](SECURITY.md#password-recovery) for the residual risks.

No database migration is needed: this uses the existing `PasswordResetToken`
table and account-deletion cascade. Configure the relay first, deploy the
application, and verify delivery with a dedicated non-health-data account.
Rolling the application back removes recovery UI but does not require a schema
rollback; outstanding links expire after 30 minutes.

## Database migrations

Prisma Migrate is the only schema-change mechanism used in this repository (`prisma/migrations/`, applied in timestamp order; never edit an applied migration — add a new one).

CI replays the migrations twice over: the integration job runs `prisma migrate deploy` into a fresh database, then diffs the result against `prisma/schema.prisma` (`prisma migrate diff --exit-code`, using a separate empty shadow database). A schema edit with no matching migration therefore fails CI instead of surfacing in production.

- **`npm run db:migrate`** (`prisma migrate dev`) — development only. Creates a new migration from any schema diff, applies it, and regenerates the Prisma Client. Needs a database it can create a disposable shadow database against.
- **`npm run db:deploy`** (`prisma migrate deploy`) — the production-safe command. Applies any pending migrations in order, does not create or need a shadow database, and does not prompt. This is what any deploy pipeline should run.
- **`npm run db:push`** (`prisma db push`) — schema-sync without a migration file. Useful for local prototyping only; never use it against a database with data you care about, since it can't be rolled back the way a migration can.

`prisma/schema.prisma` declares two connection strings on the `datasource` block: `url = env("DATABASE_URL")` and `directUrl = env("DIRECT_DATABASE_URL")`. Migrations always run against `directUrl` — see the pooled-vs-direct explanation below.

## Deploying to Vercel, step by step

1. **Provision Postgres.** Use any managed Postgres provider that gives you both a pooled and a direct connection string (Vercel Postgres, Neon, Supabase, and Railway all do). If your provider only exposes one connection string, put the same value in both `DATABASE_URL` and `DIRECT_DATABASE_URL` — it will work, but you lose the benefit of connection pooling under load, since Vercel's serverless functions each open their own connection.
2. **Set environment variables** on the Vercel project (Project Settings → Environment Variables): `DATABASE_URL` (pooled), `DIRECT_DATABASE_URL` (direct/non-pooled), `AUTH_SECRET` (a fresh, high-entropy value — never reuse a development one), and optionally `AI_PROVIDER`/`ANTHROPIC_API_KEY`/`ANTHROPIC_MODEL`/`OPENAI_API_KEY`/`OPENAI_MODEL`/`NEXT_PUBLIC_APP_URL`. Set them for the Production environment, and for Preview if you want preview deploys to hit a database (see step 4 for what that implies).
3. **Build command**: `vercel.json` overrides the default with `prisma generate && if [ "$VERCEL_ENV" = "production" ]; then prisma migrate deploy; fi && next build`. `prisma generate` has to run in the build because the Prisma Client is generated from `prisma/schema.prisma` into `node_modules/@prisma/client` and is not committed to the repo — a fresh Vercel build container has no client to import until this runs. `postinstall` also runs `prisma generate`, so `npm ci` on Vercel generates it a second time; both are safe and idempotent.
4. **Migrations run in the Production build, and only there.** The `VERCEL_ENV=production` guard in `buildCommand` means a push to the production branch applies pending migrations (through `DIRECT_DATABASE_URL`, which therefore must be set for the Production environment) before `next build`, and a Preview build never migrates. The guard exists because Vercel builds every pushed branch as a Preview: without it, an unreviewed pull request's migration would be applied to whichever database the Preview environment points at, before review or merge. What this implies:
   - **A Preview deployment of a pull request that adds a migration runs against a schema that does not have it.** Point Preview at a throwaway database (a Neon branch per preview is the natural fit; whether one is configured cannot be told from this repository) and migrate that database yourself, or accept that such previews error on the new tables. Do not point Preview at the production database and migrate it by hand to make a preview work.
   - **The migration happens before the build finishes.** If `next build` then fails, the production database is already ahead of the code still being served. Migrations must therefore be backward compatible with the previous release (additive, nullable, expand-then-contract), as `AGENTS.md` already requires for schema changes.
   - **Two production builds racing.** Prisma Migrate takes a database advisory lock, so concurrent `migrate deploy` runs serialize or one fails with a lock timeout rather than interleaving; a failed run fails that build, and the next build retries. This has not been exercised against this project's Vercel setup.
   - **A migration failure fails the build**, so the previous deployment keeps serving. Fix forward with a new migration; never edit an applied one.
   - **Not on Vercel?** Run `DATABASE_URL=$DIRECT_DATABASE_URL npx prisma migrate deploy` from your own pipeline before promoting the deployment, using the **direct**, non-pooled URL: Prisma Migrate issues session-level DDL that a transaction-mode pooler (PgBouncer, Vercel's pooled Postgres) cannot execute.
5. **Connect your GitHub repo and deploy.** Vercel builds on every push; only the production branch's build migrates. Promote Preview → Production the normal Vercel way.

## Managed PostgreSQL options

Any standard PostgreSQL works — the schema (`prisma/schema.prisma`) uses no vendor-specific extensions. Options that specifically offer the pooled + direct connection pair this app's config expects:

| Provider                        | Pooled connection               | Direct connection                                             |
| ------------------------------- | ------------------------------- | ------------------------------------------------------------- |
| Vercel Postgres                 | Yes (built-in, PgBouncer-based) | Yes                                                           |
| Neon                            | Yes (`-pooler` hostname suffix) | Yes (non-pooler hostname)                                     |
| Supabase                        | Yes (port 6543, PgBouncer)      | Yes (port 5432)                                               |
| Railway / self-managed Postgres | No pooling by default           | Use the same URL for both, or put PgBouncer in front yourself |

## Environment and secrets management

- Never commit a real `.env` — `.env.example` is the checked-in template with placeholder/local values only.
- `AUTH_SECRET` must be at least 32 characters or `lib/auth/session.ts` throws at first use (`secret()`); generate a fresh one per environment, never reuse a development or preview value in production.
- Dependency advisories: `npm run audit` (`npm audit --omit=dev --audit-level=high`) runs in CI and gates merges. At the time of writing it reports no vulnerabilities in production dependencies. Dev-dependency advisories do not gate merges; Dependabot (`.github/dependabot.yml`) opens weekly grouped PRs for npm, GitHub Actions and the `ml/` pip requirements. If a transitive advisory ever has no fix, record it here with the reason it is unreachable rather than adding `continue-on-error` to the step.
- Keep `ANTHROPIC_API_KEY`/`OPENAI_API_KEY` out of Preview environments unless you specifically want preview deployments calling a paid external API with real usage.
- `AI_PROVIDER` defaults to `local` if unset — a fresh deployment with no AI environment variables configured at all still fully works, just without a hosted LLM (see [AI_ARCHITECTURE.md](AI_ARCHITECTURE.md)).

## Monitoring and health checks

`app/api/health/route.ts` is a liveness/readiness probe: it runs `SELECT 1` against the database and returns `{ status: 'ok' | 'degraded', database: boolean, uptimeSeconds, checkedInMs }` — HTTP 200 when the database answers, 503 when it doesn't. The route's own comment states its intent directly: it "deliberately exposes no counts, no user information and no configuration — only what an operator needs to know." Point your uptime monitor (Vercel's own, UptimeRobot, Better Stack, etc.) at `GET /api/health`.

Beyond that endpoint, this repository does not wire up any monitoring, error tracking, or metrics export — no Sentry, no OpenTelemetry, nothing in `next.config.ts` beyond security headers.

## Continuous integration

`.github/workflows/ci.yml` runs on every pull request and on pushes to `main` (each push to `main` gets its own run keyed by commit, so a newer push never cancels or replaces it; a pull request's superseded runs are cancelled). Node comes from `.nvmrc`. Jobs: **quality** (typecheck, lint with zero warnings allowed, Prettier, unit tests, `prisma validate`, production `npm audit`), **integration** (`prisma migrate deploy`, the schema-drift check, then the integration suite against Postgres 16), **build** (`next build`, handed to e2e as an artifact), **e2e** (Playwright and axe against a prebuilt `.next`, with no retries and a step-level timeout, so the report is uploaded even when the suite hangs) and **research-pipeline** (`ml/` pytest on Python 3.11). Nothing in CI deploys; Vercel deploys from Git independently.

`seed-demo.yml` is manual-only and writes the synthetic demo account to a real Neon database; see the header of that file.

## Background jobs

There is no independent queue, worker process or scheduled/cron job. Recovery
mail uses Next's `after()` callback after the response and remains bounded by the
host request lifecycle. Every other expensive operation — the full analytics engine (`lib/analytics/engine.ts`'s `runAnalytics()`: summary stats, associations, anomaly detection, trend detection, clustering, feature importance), file import parsing, and every AI call — runs synchronously inside the request that triggered it (a page render or a Server Action). Apart from that `after()` callback there is no queue, no worker process, and no scheduled/cron job anywhere in this codebase.

This is fine at the data volumes a single user's health log produces (the analytics window defaults to 30–90 days of records, not years of raw CGM data), but it is a real constraint to know about before scaling:

- A slow analytics run blocks the page render or the assistant response it's part of — there's no "compute now, show results when ready" pattern.
- A large file import (`lib/import/limits.ts` caps at 20 MB) is parsed entirely within one request/response cycle, bounded by your platform's function execution timeout (Vercel's serverless function limit, unless you're on a plan with longer limits).
- If you outgrow synchronous processing — very large imports, or wanting to pre-compute analytics on a schedule rather than per-request — the natural seam to introduce a queue is `lib/services/import-service.ts`'s `commitImport()` (import) and `lib/services/analytics-service.ts`'s `analyzeUser()` (analytics): both are already isolated, side-effect-scoped functions that a background worker could call instead of a request handler, without changing their signatures. No such worker exists today.
