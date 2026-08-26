# CLAUDE.md

Guidance for Claude Code when working in this repository.

DiaLog is an accessible glucose and metabolic health tracking app: Next.js 15
(App Router), React 19, TypeScript, Tailwind v4, Prisma + Postgres, with a
separate Python research pipeline in `ml/`.

It handles personal health data. That single fact drives most of the invariants
below — when a rule here looks over-cautious, it is deliberate.

## Commands

```bash
npm run dev              # dev server (port 3000)
npm run typecheck        # tsc --noEmit
npm run lint             # eslint
npm run format:check     # prettier --check  (CI fails on this; run format first)
npm run format           # prettier --write
npm test                 # unit suite only — no database needed
npm run test:integration # integration + security suite — needs a real Postgres
npm run test:e2e         # Playwright e2e + axe accessibility (serves on port 3100)
npm run build            # prisma generate && next build
npm run db:migrate       # prisma migrate dev
npm run db:seed          # tsx prisma/seed.ts
```

`postinstall` runs `prisma generate`, so a fresh `npm ci` leaves a usable
client. After editing `prisma/schema.prisma`, re-run `prisma generate`.

While iterating, run the narrowest check that proves the change — a single
vitest file beats the whole suite. Run the full suite once before handing work
back.

## Layout

| Path             | What it owns                                                                                                                                                |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `app/`           | Routes. `(marketing)/` public pages, `(auth)/` sign-in & sign-up, `app/` the authenticated product, `api/` route handlers (auth callbacks, export, health). |
| `components/`    | `ui/` design-system primitives, `auth/`, `charts/`.                                                                                                         |
| `lib/actions/`   | Server Actions — the mutation path for the app UI.                                                                                                          |
| `lib/services/`  | Orchestration between actions/routes and the lower-level libs.                                                                                              |
| `lib/import/`    | `parse` → connector `parse()` → `normalize` → `dedupe` → `summary`. `connectors/` holds one file per device/vendor.                                         |
| `lib/analytics/` | `engine.ts` orchestrates `glucose`, `associations`, `stats` and `ml/` (anomaly, cluster, trend, importance) into an `AnalyticsResult`.                      |
| `lib/ai/`        | Provider abstraction (`providers/` local, anthropic, openai), prompts, Zod schemas, `guardrails.ts`, `redact.ts`.                                           |
| `lib/auth/`      | JWT session cookie, bcrypt passwords, rate limiting, audit log, `oauth/` (Google).                                                                          |
| `lib/domain/`    | `units` (mg/dL ↔ mmol/L), `thresholds`, `evidence` grading, `time`, `dedupe`.                                                                              |
| `lib/db/`        | Prisma client singleton and health-record query helpers.                                                                                                    |
| `prisma/`        | Schema, migrations, seed.                                                                                                                                   |
| `ml/`            | Standalone Python project (pytest). Independent of the Node build.                                                                                          |
| `docs/`          | ARCHITECTURE, AI_ARCHITECTURE, SECURITY, ACCESSIBILITY, ML_PIPELINE, DEPLOYMENT, DEVICE_INTEGRATIONS. Read the relevant one before a large change.          |

## Invariants

These encode past bugs and safety decisions. Do not relax one without saying so
explicitly.

**Units.** Storage is always canonical: glucose in mg/dL, mass in kg, volume in
mL, duration in minutes. Display unit is a per-user preference. All conversion
goes through `lib/domain/units.ts` — never convert inline in a component.

**Auth.** `middleware.ts` is an edge fast-reject for `/app/**` only; it verifies
the signature and expiry but cannot read `tokenVersion` (no DB at the edge). The
real authorization boundary is `requireUser()` server-side. Middleware must
never redirect a signed-in-looking visitor away from `/sign-in` — that produced
an infinite redirect loop that locked out users revoked via "sign out
everywhere". Bumping `User.tokenVersion` is what invalidates outstanding
cookies. `AUTH_SECRET` must be ≥32 chars; it fails closed by design.

**Import.** Connectors are pure transforms — they never touch the database, and
they never silently drop a row: anything not convertible to a `NormalizedRecord`
must produce a `RowIssue`. Register a new one by adding it to the `CONNECTORS`
array in `lib/import/connectors/registry.ts`; order is the tie-break, so
specific connectors go before generic fallbacks. `dedupeKey` uniqueness is what
makes re-importing the same export a no-op — don't bypass it.

**AI.** The AI layer never sees raw health records — only the evidence-graded
`AnalyticsResult` / `EvidenceBundle`. Provider resolution is explicit arg →
`AI_PROVIDER` → `local`, falling back to `local` when the chosen provider is
unavailable. `redact.ts` strips free-text before anything reaches a provider
with `isExternal: true` without consent, so route user notes into
`EvidenceBundle.summary` under a well-known key rather than a bespoke field.
The medical-safety regexes in `guardrails.ts` are intentionally over-broad — a
false rejection falls back to a safe template, a false negative is a dosing
instruction reaching a patient. Do not "fix" apparent over-matching without
weighing that.

**Analytics.** Everything is evidence-graded so nothing overclaims from a small
sample. `stats.ts` primitives return `null`, never `NaN` or `Infinity`, when a
result is undefined.

**Tests.** Never skip, disable, or quarantine a test to get green. The
integration suite hard-refuses any `DATABASE_URL` not naming `dialog_test` —
that guard protects real databases, so don't work around it.

## Testing

- **Unit** (`tests/unit/`) — pure logic, no DB. This is `npm test`.
- **Integration** (`tests/integration/`) — real Postgres. Runs single-forked and
  sequentially on purpose: every file shares one physical database.
  `setup-env.ts` overrides the connection URL before Prisma is imported.
- **E2E** (`tests/e2e/`) — Playwright + axe, served on **port 3100**. Runs
  `workers: 1`, `fullyParallel: false` on purpose: the specs share a small pool
  of accounts (sign-up is rate limited) and several assert on record counts.

E2E specifics worth knowing before you touch that suite: `global-setup.ts`
resets and reseeds the DB; `auth.setup.ts` pre-authenticates the shared accounts
once into `tests/e2e/.auth/*.json`; `fake-google.ts` is a loopback OIDC stand-in
so the OAuth flow is testable without real Google. Don't add new independent
sign-up flows casually — the rate-limit budget is shared. And `locator.count()`
is blocked by an ESLint rule: it doesn't auto-wait and reads 0 against streamed
pages. Use `countRecordRows()` from `tests/e2e/records.ts`.

## Style

The codebase writes thorough "why" comments, frequently naming the specific bug
a decision came from. Match that: explain the non-obvious reason, not the
obvious mechanic. Files are kebab-case, one connector or provider per file. Zod
handles structured-output and input validation. The AI layer throws a typed
`AIProviderError` with a `kind` enum rather than raw errors.

## Environment

`.env.example` lists what's needed: `DATABASE_URL`, `DIRECT_DATABASE_URL`,
`AUTH_SECRET`, `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `AI_PROVIDER`,
`ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`, `OPENAI_API_KEY`, `OPENAI_MODEL`,
`NEXT_PUBLIC_APP_URL`. Never commit real values, and never paste a secret into a
subagent brief, a commit message, or a PR body.

## Skills

Vendored under `.claude/skills/` and committed, so every session and every
contributor gets the same behaviour with no install step. See
`.claude/skills/README.md` for the full table and the quality bar for adding
more.

Reach for them rather than improvising:

- `frontend-design`, `ui-ux-pro-max` — building or reshaping UI.
- `web-design-guidelines` — reviewing UI code before you call it done.
- `accessibility` — WCAG 2.2 auditing. This is an accessibility-first product
  with an axe-core e2e suite; treat this skill as part of the definition of
  done for any UI change, not an optional extra.
- `vercel-react-best-practices` — React 19 / Next.js 15 performance work.
- `prisma-cli` — migrations, generate, seeding, studio.
- `find-skills` — before hand-rolling a capability that probably already exists
  as a skill.

## Agent workflow

### Orchestrate, don't do everything yourself

The default posture for anything beyond a one-file change is: **the top-level
session plans and reviews, subagents execute.** The orchestrator's context is
the scarcest resource in the session — every file a subagent reads is a file the
orchestrator never has to hold.

Plan first, in the orchestrator. Read only enough to decide _what_ needs to
happen and _which files are involved_, then hand each unit of work to an agent
with a brief that already contains the facts you established, so the subagent
never re-derives them.

### Picking a model and effort level

Match the model to the shape of the work, not to how important it feels:

| Work                                                                                                                                                                               | Model                       | Notes                                                                            |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------- | -------------------------------------------------------------------------------- |
| Planning, architecture, cross-cutting refactors, security-sensitive changes (`lib/auth/**`, `middleware.ts`), final review                                                         | **Opus** — the orchestrator | Keep these in the top-level session.                                             |
| A scoped feature, a bug fix inside one module, a new import connector, a CI/config change, writing a test suite                                                                    | **Sonnet** subagent         | The workhorse. Most delegated work lands here.                                   |
| Mechanical, verifiable, low-judgement work — renames, adding a fixture, mapping one file's exports, running a command and reporting output, grepping for a pattern across the tree | **Haiku** subagent          | Cheap and fast. If a task needs no design judgement, it should not be on Sonnet. |

Effort follows the same logic. Spend it where a wrong answer is expensive:

- **High effort** — anything touching auth, session handling, authorization
  checks, glucose-value units, or dedupe logic. A silent bug there is a
  correctness or safety problem, not a cosmetic one.
- **Medium effort** — ordinary feature and fix work, test authoring, CI.
- **Low effort** — lookups, inventories, formatting, "what does this file
  export", "does this pattern appear anywhere else".

Use the `Explore` agent for broad read-only fan-out (searching many files,
"where is X handled") — it returns the conclusion instead of dumping file
contents into the orchestrator's context. Use `Plan` when the shape of a change
is genuinely unclear before committing to it.

### Writing a good subagent brief

A subagent starts cold. A brief that omits context guarantees the subagent
re-reads what you already read, which costs more than writing the brief. Every
brief should carry:

1. **Exact scope** — the files it may touch, and the files it must not.
2. **Facts you already established** — paths, function names, the failing
   command and its output, the constraint you discovered. Paste them; do not
   make the agent go find them.
3. **The invariants below** that apply, restated. Do not assume it will read
   this file.
4. **Verification** — the exact command it must run before reporting done
   (`npm run typecheck`, `npm test`, the specific vitest file).
5. **"Do not commit or push."** The orchestrator owns the git history so that
   one reviewer sees one coherent diff.
6. **What to report back** — a short structured summary, not a transcript.

Run independent subagents in parallel in a single message. Serialize only where
one genuinely needs another's output.

### Parallelism and conflicts

Two agents editing the same file will clobber each other. Partition by
directory or by file before dispatching. When two units of work genuinely
overlap, either merge them into one brief or run them in sequence. For work
that needs to build and test in isolation, give the agent its own worktree.

### Reviewing what comes back

Subagent output is a claim, not a result. Before accepting it: read the diff,
run the verification command yourself, and check it did not widen scope. A
subagent reporting "all tests pass" without the output is not evidence. If the
work is wrong, send a correction to the same agent — it still has its context —
rather than spawning a fresh one.

## Token efficiency

The point of all of the above is to keep the orchestrator's context small enough
to stay accurate over a long session. Concretely:

- **Never `cat` a large file to see a small part of it.** Use `rg` with context,
  or `sed -n 'START,ENDp'`. `package-lock.json` (340KB), `README.md` (37KB) and
  `.claude/skills/ui-ux-pro-max/data/**` must never be read whole.
- **Search before reading.** `rg -n 'pattern'` to locate, then read the range.
  Prefer `rg --files-with-matches` when you only need to know where something
  lives.
- **Delegate wide reads.** If answering a question means opening more than a
  handful of files, that is an `Explore` agent's job, not the orchestrator's.
- **Don't re-read a file you just edited.** The edit tools fail loudly; a
  successful edit needs no confirmation read.
- **Batch independent tool calls** into one message rather than round-tripping.
- **Run the narrowest check that proves the change.** A single vitest file
  (`npx vitest run tests/unit/import/dedupe.test.ts`) beats the whole suite
  while iterating; run the full suite once at the end.
- **Prefer targeted commands over exploratory ones.** `git diff --stat` before
  `git diff`. `npm run typecheck` before a full build.
- **Summarize, don't paste.** When reporting to the user or to another agent,
  give the conclusion and the file:line, not the file contents.
