# Deployment

## Local development

1. Install Node.js ≥ 20 and a local PostgreSQL instance.
2. `npm install` (runs `prisma generate` via `postinstall`).
3. `cp .env.example .env` and fill in `DATABASE_URL` / `DIRECT_DATABASE_URL` (point both at the same local database for development — there is no pooler to distinguish them from) and `AUTH_SECRET` (generate with `node -e "console.log(require('crypto').randomBytes(48).toString('base64url'))"`).
4. `npx prisma migrate deploy` to apply migrations (or `npm run db:migrate` — `prisma migrate dev` — if you intend to author a new migration).
5. `npm run db:seed` to populate the demo account (`demo@dialog.health` / `demo-account-2026`) with three months of synthetic data.
6. `npm run dev` and visit `http://localhost:3000`.

See the README's [Quick start](../README.md#quick-start) for the same steps with more explanation, and [Environment variables](../README.md#environment-variables) for the full table.

## Database migrations

Prisma Migrate is the only schema-change mechanism used in this repository (`prisma/migrations/`, currently one migration: `20260824201633_init`).

- **`npm run db:migrate`** (`prisma migrate dev`) — development only. Creates a new migration from any schema diff, applies it, and regenerates the Prisma Client. Needs a database it can create a disposable shadow database against.
- **`npm run db:deploy`** (`prisma migrate deploy`) — the production-safe command. Applies any pending migrations in order, does not create or need a shadow database, and does not prompt. This is what any deploy pipeline should run.
- **`npm run db:push`** (`prisma db push`) — schema-sync without a migration file. Useful for local prototyping only; never use it against a database with data you care about, since it can't be rolled back the way a migration can.

`prisma/schema.prisma` declares two connection strings on the `datasource` block: `url = env("DATABASE_URL")` and `directUrl = env("DIRECT_DATABASE_URL")`. Migrations always run against `directUrl` — see the pooled-vs-direct explanation below.

## Deploying to Vercel, step by step

1. **Provision Postgres.** Use any managed Postgres provider that gives you both a pooled and a direct connection string (Vercel Postgres, Neon, Supabase, and Railway all do). If your provider only exposes one connection string, put the same value in both `DATABASE_URL` and `DIRECT_DATABASE_URL` — it will work, but you lose the benefit of connection pooling under load, since Vercel's serverless functions each open their own connection.
2. **Set environment variables** on the Vercel project (Project Settings → Environment Variables): `DATABASE_URL` (pooled), `DIRECT_DATABASE_URL` (direct/non-pooled), `AUTH_SECRET` (a fresh, high-entropy value — never reuse a development one), and optionally `AI_PROVIDER`/`ANTHROPIC_API_KEY`/`ANTHROPIC_MODEL`/`OPENAI_API_KEY`/`OPENAI_MODEL`/`NEXT_PUBLIC_APP_URL`. Set them for the Production (and Preview, if you want preview deploys to hit a database) environment.
3. **Build command**: Vercel auto-detects Next.js and runs `npm run build`, which is `prisma generate && next build` (`package.json`). `prisma generate` has to run in the build because the Prisma Client is generated from `prisma/schema.prisma` into `node_modules/@prisma/client` and is not committed to the repo — a fresh Vercel build container has no client to import until this runs. `postinstall` also runs `prisma generate`, so a plain `npm install` on Vercel generates it a second time; both are safe and idempotent.
4. **Run migrations before traffic reaches the new deployment.** This repository does not wire a migration step into the Vercel build itself (running `prisma migrate deploy` as part of the Vercel build is possible but risky if two deploys race on the same database) — the recommended path is a separate deploy-pipeline step: `DATABASE_URL=$DIRECT_DATABASE_URL npx prisma migrate deploy` run from CI (or Vercel's "Deploy Hooks" combined with a small CI job) before promoting the deployment, using the **direct**, non-pooled URL, since Prisma Migrate issues session-level DDL that a transaction-mode pooler (PgBouncer, Vercel's pooled Postgres) cannot execute.
5. **Connect your GitHub repo and deploy.** Vercel builds on every push; promote Preview → Production the normal Vercel way once migrations for that change have been applied.

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
- Keep `ANTHROPIC_API_KEY`/`OPENAI_API_KEY` out of Preview environments unless you specifically want preview deployments calling a paid external API with real usage.
- `AI_PROVIDER` defaults to `local` if unset — a fresh deployment with no AI environment variables configured at all still fully works, just without a hosted LLM (see [AI_ARCHITECTURE.md](AI_ARCHITECTURE.md)).

## Monitoring and health checks

`app/api/health/route.ts` is a liveness/readiness probe: it runs `SELECT 1` against the database and returns `{ status: 'ok' | 'degraded', database: boolean, uptimeSeconds, checkedInMs }` — HTTP 200 when the database answers, 503 when it doesn't. The route's own comment states its intent directly: it "deliberately exposes no counts, no user information and no configuration — only what an operator needs to know." Point your uptime monitor (Vercel's own, UptimeRobot, Better Stack, etc.) at `GET /api/health`.

Beyond that endpoint, this repository does not wire up any monitoring, error tracking, or metrics export — no Sentry, no OpenTelemetry, nothing in `next.config.ts` beyond security headers. `.github/workflows/ci.yml` in this repository is a leftover from the project's earlier Python-only prototype (it sets up Python 3.9/3.10/3.11 and runs `pytest`) and does **not** currently build, lint, typecheck, or test the Next.js/TypeScript application — running `npm run lint`, `npm run typecheck`, `npm test`, and (with a database available) the integration suite in CI is something you would need to add, not something already wired up.

## Background jobs

There are none. Every expensive operation — the full analytics engine (`lib/analytics/engine.ts`'s `runAnalytics()`: summary stats, associations, anomaly detection, trend detection, clustering, feature importance), file import parsing, and every AI call — runs synchronously inside the request that triggered it (a page render or a Server Action). There is no queue, no worker process, and no scheduled/cron job anywhere in this codebase.

This is fine at the data volumes a single user's health log produces (the analytics window defaults to 30–90 days of records, not years of raw CGM data), but it is a real constraint to know about before scaling:

- A slow analytics run blocks the page render or the assistant response it's part of — there's no "compute now, show results when ready" pattern.
- A large file import (`lib/import/parse.ts` caps at 100 MB) is parsed entirely within one request/response cycle, bounded by your platform's function execution timeout (Vercel's serverless function limit, unless you're on a plan with longer limits).
- If you outgrow synchronous processing — very large imports, or wanting to pre-compute analytics on a schedule rather than per-request — the natural seam to introduce a queue is `lib/services/import-service.ts`'s `commitImport()` (import) and `lib/services/analytics-service.ts`'s `analyzeUser()` (analytics): both are already isolated, side-effect-scoped functions that a background worker could call instead of a request handler, without changing their signatures. No such worker exists today.
