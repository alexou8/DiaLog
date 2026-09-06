# Resume and portfolio copy

Reusable descriptions of DiaLog, written to be pasted into a resume, a
portfolio site, or a LinkedIn project entry.

**Every claim below is backed by something in this repository.** Where a number
would need to be measured rather than counted, it is left as a bracketed
placeholder — fill it in from a real measurement or delete the clause. Do not
substitute an estimate.

Specifically, do **not** add: user counts, adoption or engagement figures,
clinical outcomes or validation, compliance certifications (HIPAA, PIPEDA,
PHIPA, FDA, Health Canada), uptime figures, or performance percentages that
have not been measured on a fixed dataset. None of those exist for this
project.

---

## 25-word description

> An accessible glucose and metabolic health tracker that grades every insight
> by the evidence behind it, and says "not enough data yet" when that is true.

## 50–75 word portfolio description

> DiaLog is a personal glucose and metabolic health record built with Next.js
> 15, TypeScript and PostgreSQL. It imports readings from vendor and device
> export files, grades every statistical finding by the sample size behind it,
> and explains patterns in plain language through an assistant that never
> receives a raw health record — only a pre-aggregated evidence bundle. Built
> accessibility-first for users aged 20 to 80, with axe checks in CI.

## Resume bullets

Action + implementation + verified outcome. Pick two or three; they are ordered
by how much engineering they demonstrate.

> Engineered a multi-format health-data ingestion pipeline — six vendor
> connectors plus three generic fallbacks over CSV, XLSX, JSON and XML — with
> preview-then-commit, content-addressed deduplication that makes re-import a
> no-op, and per-row issue reporting instead of silent drops; verified against
> real export fixtures plus malformed-file and re-import cases across unit,
> integration and browser tests.

> Designed a privacy boundary that lets an AI assistant explain personal health
> data without ever receiving it: the provider layer is handed only a
> pre-aggregated, evidence-graded bundle, so a local deterministic provider and
> an external API receive identical input, with schema validation and
> deliberately over-eager medical-safety filters that fall back to a safe
> template rather than risk emitting dosing language.

> Built the authentication and account lifecycle — signed-cookie sessions,
> bcrypt credentials, Google OAuth, session revocation, per-user data
> isolation, audit events, export and deletion — and enforced its hardest
> invariant with a build-failing test that rejects session-revocation writes
> from the wrong architectural layer, after that mistake produced a redirect
> loop locking out the users who had just revoked their sessions.

> Made accessibility a build gate rather than an audit: WCAG 2.2 AA target, a
> real `<table>` alternative behind every hand-built SVG chart, no meaning
> encoded in colour alone, and `@axe-core/playwright` running over twelve
> public and authenticated routes on every CI run.

## Technology line

> Next.js 15 (App Router) · React 19 · TypeScript (strict) · PostgreSQL ·
> Prisma · Zod · Tailwind CSS v4 · Radix · Vitest · Playwright · axe-core ·
> GitHub Actions

## Interview talking points

1. **"Design an AI feature that cannot leak the data it reasons about."** The
   answer is architectural rather than prompt-level, and the honest cost is
   that the assistant can only answer questions the analytics layer already
   models. See [CASE_STUDY.md](CASE_STUDY.md).
2. **"When is a test that reads source code the right tool?"** The
   session-revocation guardrail expresses an invariant about _where_ code lives
   that the type system cannot — and it exists because the alternative already
   failed in production-shaped ways.
3. **"How do you build analytics that are allowed to return nothing?"** Sparse,
   self-reported, irregular data means the honest output is often "no finding";
   designing for an absence as the successful state was harder than the
   statistics.

## Placeholders to fill in only from real measurement

- `[measured p95 for the analytics request path on an N-record dataset]` — no
  benchmark has been run; do not claim a performance figure until one has.
- `[live demo URL]` — only if a healthy public deployment exists.
- Test counts — prefer linking the CI run over pasting a number that will
  drift.
