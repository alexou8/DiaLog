# DiaLog — case study

A portfolio write-up of why DiaLog is built the way it is. The product itself
is described in [PRD.md](PRD.md); the system boundaries are in
[ARCHITECTURE.md](ARCHITECTURE.md). This document covers the decisions and the
reasoning behind them, including the options that were rejected.

## The problem

Glucose data is abundant and almost useless in the form people receive it.

A meter or CGM app will happily show someone 90 days of readings, a 14-day
average, and a percentage in range. What it does not do is answer the question
the person actually has, which is some version of _"is what I changed last
month working?"_ Answering that means comparing periods, controlling for the
obvious confounders, and — critically — knowing when the honest answer is
"there isn't enough data to say."

The two common failure modes in this space are opposite and both bad. Consumer
health apps overclaim: a five-day trend becomes a confident statement about the
user's metabolism. Clinical tools underclaim: they present raw numbers and
leave every inference to the user, which works for an endocrinologist and not
for the person living with the condition.

DiaLog's premise is that the interesting engineering problem is the honesty
constraint, not the charting.

## Intended users

People managing prediabetes or type 2 diabetes, and people who track glucose
without a diagnosis. The age range is deliberately wide — roughly 20 to 80 —
because that is the actual demographic of the condition, and it is what forced
accessibility to be a design input rather than an audit at the end.

## Scope and role

Solo project. I designed the product, the data model, the analytics approach
and the visual system, and wrote the application, the test suites, the CI
pipeline and the documentation set.

## Constraints that shaped the design

1. **It handles personal health data.** Every design decision that could leak,
   retain, or mis-scope a health record is a safety decision, not a
   correctness one.
2. **It must not practise medicine.** No diagnosis, no dose calculation, no
   interpretation that implies either. This has to be enforced in code, because
   a policy in a README does not stop a language model.
3. **The data is sparse, irregular and self-reported.** People forget to log.
   Meters get replaced. A month may have four readings or four hundred.
   Anything that assumes a regular time series is wrong.
4. **Vendor exports are inconsistent.** Same vendor, different software
   version, different column order, different date format, occasionally a
   different unit — with no version marker in the file.
5. **Accessibility is a requirement of the user base**, not a compliance
   checkbox.

## Key decisions

### Evidence grading instead of confidence styling

**Decision.** Every analytic finding carries the sample size it was computed
from, graded against thresholds in `lib/domain/evidence.ts`. Findings below the
minimum are not shown as claims at all.

**Rejected alternative.** Showing everything with a confidence percentage. It
tests well and it is dishonest: users read "62% confident" as "probably true",
and a percentage derived from six data points implies a precision the data
cannot support.

**Consequence.** The statistics primitives in `lib/analytics/stats.ts` return
`null` rather than `NaN` or `Infinity` when a result is undefined, so an
under-powered result cannot silently propagate into a rendered claim.

### A privacy boundary between health data and the AI layer

**Decision.** `lib/ai/` is only ever handed an `AnalyticsResult` /
`EvidenceBundle`: pre-aggregated, evidence-graded, no raw records. The local
deterministic provider and an external API provider receive exactly the same
shape of input.

**Rejected alternative.** Passing recent records into a prompt and letting the
model summarise. Simpler, better prose, and it puts a person's health record
into a third-party API on every request.

**Consequence.** The provider abstraction is honest by construction — turning
on an external provider changes who computes the sentence, not what data
leaves the deployment. `lib/ai/redact.ts` strips free text before anything
reaches a provider marked `isExternal` without consent.

### Guardrails that are deliberately over-eager

**Decision.** The medical-safety regexes in `lib/ai/guardrails.ts` over-match,
and a rejection falls back to a safe template.

**Reasoning.** The two error types are not symmetric. A false positive costs a
user one slightly stilted sentence. A false negative is a dosing instruction
reaching a patient. Tuning for precision here optimises the wrong metric.

### Session revocation enforced by a test that reads source code

**Decision.** `User.tokenVersion` is what invalidates outstanding session
cookies, and it may only be written from a route handler.
`tests/unit/auth/session-revocation.test.ts` scans the source text of
`lib/actions/` and fails the build if a Server Action writes that column.

**Reasoning.** This is a scar. An earlier version put revocation in a Server
Action, and a revoked device hit `ERR_TOO_MANY_REDIRECTS` and could never
recover on its own — the users locked out were exactly the ones who had just
pressed "sign out everywhere" because they thought their account was
compromised. The rule that came out of it is written in `docs/SECURITY.md`: _an
unverifiable token may be used to deny access to a protected route, never to
deny access to the recovery page._

**Rejected alternative.** A code comment. Comments do not fail builds.

### Content-addressed deduplication instead of import bookkeeping

**Decision.** Each normalized record gets a `dedupeKey` derived from its own
content. Re-importing the same export is a no-op because the keys collide.

**Rejected alternative.** Tracking imported files by name or hash. It breaks
the moment someone exports an overlapping date range twice, which is the single
most common real-world import pattern.

**Consequence.** Connectors can be pure functions — they never touch the
database — which is why every one of them is unit-testable against a real
fixture file with no test database at all.

### Hand-built SVG charts

**Decision.** No charting library. Charts are inline SVG, and every one ships a
`<table>` alternative carrying the same data.

**Rejected alternative.** Recharts or Chart.js. Both render to canvas or to SVG
that is opaque to a screen reader, and retrofitting an accessible alternative
costs more than drawing the chart.

## Quality strategy

The suites are separated by what they need, so the fast one stays fast:

| Layer       | Needs         | Covers                                                                     |
| ----------- | ------------- | -------------------------------------------------------------------------- |
| Unit        | Nothing       | Domain rules, analytics, AI guardrails and schemas, every import connector |
| Integration | Real Postgres | Every record type round-tripped, cross-user isolation, API security, auth  |
| E2E         | Browser + DB  | Real user journeys, plus `@axe-core/playwright` over twelve routes         |

Two details worth calling out, because both came from debugging rather than
from planning:

- The integration suite **hard-refuses any `DATABASE_URL` not naming
  `dialog_test`.** The guard exists so a mistyped environment variable cannot
  point a destructive test run at a real database.
- The e2e suite runs `workers: 1, fullyParallel: false` **on purpose.** The
  specs share a small pool of accounts because sign-up is rate limited, and
  several assert on record counts. `locator.count()` is blocked by a custom
  ESLint rule after a CI-only flake was traced to it: it does not auto-wait and
  reads `0` against a streamed page.

## Limitations

Stated plainly, because a portfolio project that claims to be finished is less
credible than one that knows what it isn't:

- **Rate limiting is per-instance and in-memory.** It is correct on a single
  instance and weakens under horizontal scaling. Replacing it with a shared
  store is the next production task.
- **There is no email verification concept.** Addresses are unverified, which
  bounds what account recovery can promise.
- **There is no live device sync.** Every integration is file import. The
  documentation says "import" everywhere for that reason.
- **Analytics are computed synchronously per request.** No measurement has yet
  shown this to be a problem at realistic dataset sizes, so it has not been
  changed — but it has also not been proven safe at scale.
- **Encryption at rest is deployment-dependent**, and there is no public
  production deployment with observability, an uptime target, or an incident
  history behind it.
- **The `ml/` pipeline is offline research.** It is not deployed, not imported
  by the app, and its synthetic-data results carry no clinical validity.

## Interview discussion guide

Three problems worth talking through, and what I would change with more scale.

**1. How do you stop an AI feature from leaking health data?**
The answer that matters is architectural, not prompt engineering: make it
impossible for the raw data to reach the provider. The interesting follow-up is
what you give up — the assistant cannot answer questions the analytics layer
did not anticipate, and every new capability means extending the evidence
bundle rather than widening the prompt.

**2. When is it correct to build a test that reads source code?**
Almost never, and the session-revocation guardrail is the exception that
explains the rule: the invariant is about _where_ code lives, which no type
system in this stack can express. Discussion: what would have to be true for
this to become a lint rule or a module boundary instead.

**3. How do you design analytics that are allowed to say nothing?**
Sparse self-reported data breaks the usual assumptions, and the honest output
is frequently "no finding". Designing a UI whose most common successful state
is an absence — without it reading as a failure or an empty state — was harder
than the statistics.

**At greater scale, I would change:** the rate limiter first (shared store,
documented fail-open/fail-closed per action sensitivity), then measure the
analytics request path properly before deciding whether it needs a job boundary
or persisted insight snapshots. I would not add features until there is
production observability to tell me which ones are actually used.
