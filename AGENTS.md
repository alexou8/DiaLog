# AGENTS.md

The engineering contract for this repository. It is tool-neutral and
self-contained: Claude Code, Codex, Cursor, Copilot and any other coding agent
can work from this file alone.

**This is the single source of truth for how to work here.** Claude Code users
additionally get [CLAUDE.md](CLAUDE.md), which covers only Claude-specific
orchestration (Codex delegation, subagents, model selection, skills) and does
not repeat anything below. Agents that read `AGENTS.md` are not missing
anything essential.

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

Run these through `npm run`, never a bare `npx prettier` / `npx eslint` /
`npx tsc`. Without `node_modules` installed, `npx` silently fetches the latest
release instead of the pinned one, and its output disagrees with CI: Prettier
3.8 pads Markdown table columns containing `↔` differently from the pinned
3.6.2, so `npx prettier --write` reports files as unformatted, "fixes" them,
and leaves the tree red under `npm ci`. Run `npm ci` first if `node_modules`
is missing.

While iterating, run the narrowest check that proves the change — a single
vitest file beats the whole suite. Run the full suite once before handing work
back.

## Layout

| Path             | What it owns                                                                                                                                                  |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `app/`           | Routes. `(marketing)/` public pages, `(auth)/` sign-in & sign-up, `app/` the authenticated product, `api/` route handlers (auth callbacks, export, health).   |
| `components/`    | `ui/` design-system primitives, `auth/`, `charts/`.                                                                                                           |
| `lib/actions/`   | Server Actions — the mutation path for the app UI.                                                                                                            |
| `lib/services/`  | Orchestration between actions/routes and the lower-level libs.                                                                                                |
| `lib/import/`    | `parse` → connector `parse()` → `normalize` → `dedupe` → `summary`. `connectors/` holds one file per device/vendor.                                           |
| `lib/analytics/` | `engine.ts` orchestrates `glucose`, `associations`, `stats` and `ml/` (anomaly, cluster, trend, importance) into an `AnalyticsResult`.                        |
| `lib/ai/`        | Provider abstraction (`providers/` local, anthropic, openai), prompts, Zod schemas, `guardrails.ts`, `redact.ts`.                                             |
| `lib/auth/`      | JWT session cookie, bcrypt passwords, rate limiting, audit log, `oauth/` (Google).                                                                            |
| `lib/domain/`    | `units` (mg/dL ↔ mmol/L), `thresholds`, `evidence` grading, `time`, `dedupe`.                                                                                |
| `lib/db/`        | Prisma client singleton and health-record query helpers.                                                                                                      |
| `prisma/`        | Schema, migrations, seed.                                                                                                                                     |
| `ml/`            | Standalone Python project (pytest). Independent of the Node build.                                                                                            |
| `docs/`          | PRD, ARCHITECTURE, DATA, SECURITY, ACCESSIBILITY, AI_ARCHITECTURE, ML_PIPELINE, DEPLOYMENT, DEVICE_INTEGRATIONS. Read the relevant one before a large change. |

## Invariants

These encode past bugs and safety decisions. Do not relax one without saying so
explicitly.

**Units.** Storage is always canonical: glucose in mg/dL, mass in kg, volume in
mL, duration in minutes. Display unit is a per-user preference. All conversion
goes through `lib/domain/units.ts` — never convert inline in a component.

**Ownership.** Any table reachable with a client-supplied id must be read and
written scoped by owner — `where: { id, userId }`, never `where: { id }` alone.
This holds for every model, not just health records. An `upsert` keyed on a
client-supplied id alone is the specific mistake that produced a cross-account
write bug in the assistant; do not reintroduce it.

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
makes re-importing the same export a no-op — don't bypass it. XML input rejects
any `DOCTYPE` internal subset: entity-expansion bombs blow up after the
file-size check passes, so the size ceiling cannot bound them.

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

**Deletion.** If you add a model storing anything derived from health data,
decide explicitly whether `deleteAllRecordsAction` clears it and whether it
cascades from `User`. Getting this wrong leaves health data behind after a user
asks for it to be erased. `AuditEvent` is the one deliberate exception: its
`userId` is nullable and its foreign key is `ON DELETE SET NULL` so a security
event outlives the account it names.

**Logging.** Never log, or put in an error message, a health value or user free
text.

**Tests.** Never skip, disable, or quarantine a test to get green. The
integration suite hard-refuses any `DATABASE_URL` not naming `dialog_test` —
that guard protects real databases, so don't work around it.

## CI failures

**Fix the cause, not the symptom.** A red check is a defect report. It gets
diagnosed and resolved at its root, the same as a bug in application code. This
is not negotiable and it is not traded against speed — a suppressed failure is a
defect that survives into `main` and costs more later than the diagnosis would
have cost now.

Work every failure in this order:

1. **Read the actual error.** The verbatim `##[error]` line and the failing step
   name, from the job log. Not the summary, not the step name alone, not a guess
   from the diff. If the log tail only shows teardown, page further back until
   you have the real line.
2. **Explain the mechanism.** State why the failure happens, specifically enough
   that you could predict which other inputs would trigger it. "It's flaky",
   "it's the environment", "CI is being weird" are not mechanisms — they are
   admissions that step 2 has not happened yet.
3. **Reproduce it**, locally or in a job you can iterate on. A fix for a failure
   you never reproduced is a guess. Where the environment is the difference
   (artifact round-trips, service containers, a clean `npm ci`), reproduce _that
   difference_ rather than the convenient local approximation.
4. **Fix the root cause.** Then confirm the original failure is gone _and_ that
   the fix did not just relocate it.
5. **Close the class, not the instance.** Ask what else the same cause could
   break, and whether the failure could have been caught earlier — by a type, a
   test, or a check that runs before CI does.

Never do these to get green:

- Skip, disable, `.skip`, quarantine, or loosen the assertion of a failing test.
- Add a retry, a bare `sleep`, or a raised timeout to paper over a race. Fix the
  race. Timeouts change only when the work genuinely got slower.
- `continue-on-error`, `|| true`, or dropping a step from the workflow.
- Re-run the job hoping for different output.
- Weaken lint or type rules rather than fixing what they caught.

**Flakiness is a root cause, not an excuse.** A test that fails intermittently
has a real defect — usually a race, shared state, or an ordering assumption.
The serialization in both the integration suite (one shared physical database)
and the e2e suite (shared rate-limited accounts, count assertions) exists
because someone did this work properly; the `locator.count()` ESLint rule exists
because someone traced a CI-only flake to its mechanism instead of retrying it.
Match that standard.

**Green locally is not green.** CI differs from a dev machine in ways that
matter: a clean `npm ci`, service containers, artifacts round-tripped between
jobs, a different Node version, no warm caches. When a change touches how CI
itself is wired, verifying the local equivalent is necessary but not sufficient
— the CI-only path is exactly where the bug will be. Say plainly which parts
you verified and which you could not.

## Testing

- **Unit** (`tests/unit/`) — pure logic, no DB. This is `npm test`.
- **Integration** (`tests/integration/`) — real Postgres. Runs single-forked and
  sequentially on purpose: every file shares one physical database.
  `setup-env.ts` overrides the connection URL before Prisma is imported. Point
  it elsewhere with `TEST_DATABASE_URL`, not `DATABASE_URL`.
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

## Reasoning about a change

- **Evidence over assumption.** The implementation is the source of truth. Where
  documentation and code disagree, the code wins — then fix the documentation in
  the same change.
- **Trace the failure paths, not just the happy path.** For any workflow you
  touch: invalid input, unauthenticated, another user's data, missing resource,
  empty state, dependency failure, partial failure, concurrent modification.
- **Prefer the simplest design that expresses the domain.** Do not add an
  abstraction for a single caller.
- **Do not delete uncertain code.** Flag it. Absence of a static import is not
  proof something is dead — check dynamic imports, Next.js convention-based
  entrypoints (`page.tsx`, `layout.tsx`, `route.ts`, `middleware.ts`), scripts,
  workflows, `public/sw.js`, seeds, and string-based references first.
- **Never weaken a security boundary to simplify code.** Security beats
  cleanliness, every time.

## Working as a delegated implementation agent

Much of the work here is dispatched by an orchestrating agent that owns
planning, review, and repository integration. If you were handed a bounded task
rather than talking to the user directly:

- **Report, do not just finish.** Return what you changed and why, the files
  touched, the design decisions and assumptions you made, anything you
  discovered about the repository that the brief got wrong, the exact
  verification commands you ran and their exact output, and what you could not
  verify. "Done." is not a report.
- **Surface conflicts instead of working around them.** If the task conflicts
  with the repository, would violate an invariant above, rests on a wrong
  assumption, duplicates an abstraction that already exists, or hides a
  security or health-data problem, say so and stop rather than implementing it
  quietly. Repository evidence outranks the brief. You are expected to push
  back, and the orchestrator decides.
- **Do not commit, push, merge, rewrite history, or open pull requests.** The
  orchestrator owns repository integration so that one reviewer sees one
  coherent diff.
- **Do not widen scope silently.** Stay inside the files you were given; if the
  fix genuinely requires more, report that instead of expanding on your own.

## Destructive and high-risk changes

- **Database.** Schema changes need a migration plus an explicit read on
  existing data, deployment order, rollback, and application compatibility.
  Additive nullable columns and foreign-key action changes are safe to ship
  alone; column removals and type changes are not. Never hand-edit an applied
  migration — add a new one.
- **Dependencies.** Every dependency must justify its existence. Pin exact
  versions, and declare anything you import directly — a package that only
  resolves transitively will break on a clean install. Check `npm audit` for
  anything reachable from user input.
- **Generated code.** Prisma client output, `next-env.d.ts` and `.next/` are
  generated. Regenerate with the repo's tooling; never edit by hand.
- **Vendored skills** under `.claude/skills/` are third-party content managed by
  the `skills` CLI. Do not hand-edit them.

## Accessibility

This is an accessibility-first product with an axe-core e2e suite. Treat WCAG
2.2 conformance as part of the definition of done for any UI change, not an
optional extra. Meaning is never encoded in colour alone; every chart ships a
real `<table>` data alternative; focus must stay visible; contrast must hold in
both light and dark themes. See [docs/ACCESSIBILITY.md](docs/ACCESSIBILITY.md).

## Verification

```bash
npm run typecheck && npm run lint && npm run format:check && npm test
```

Then, where the change warrants it: `npm run test:integration` (needs a real
Postgres named `dialog_test`), `npm run test:e2e`, `npm run build`.

A suite passing without you having seen its output is not evidence. Say plainly
which parts you verified and which you could not.

## Documentation expectations

Documentation is part of the change, not a follow-up. If you alter behaviour a
document describes, update that document in the same commit. Do not inflate: a
document must be accurate, useful, and maintainable. Do not manufacture
certainty — mark inferences as inferences.

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

## Token efficiency

Keep your working context small enough to stay accurate over a long session:

- **Never `cat` a large file to see a small part of it.** Use `rg` with context,
  or `sed -n 'START,ENDp'`. `package-lock.json`, `README.md` and
  `.claude/skills/ui-ux-pro-max/data/**` must never be read whole.
- **Search before reading.** `rg -n 'pattern'` to locate, then read the range.
- **Don't re-read a file you just edited.** The edit tools fail loudly.
- **Prefer targeted commands.** `git diff --stat` before `git diff`.
  `npm run typecheck` before a full build.
- **Summarize, don't paste.** Give the conclusion and the `file:line`.
