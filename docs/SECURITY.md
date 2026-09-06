# Security

This document describes DiaLog's actual threat model and controls, citing the files that implement each one, and closes with a checklist for anyone deploying it for real.

## Threat model

DiaLog stores personal health data (glucose readings, meals, medications, symptoms, mood) for individual users. The controls below are built around three primary risks: **account takeover** (another party reading or writing a user's health records), **cross-user data leakage** (one user's data appearing in another user's view via a missing scope check), and **health data leaving the deployment** without the user's knowledge (via an external AI provider, logs, or a cache). There is no admin role, no multi-tenant organisation model, and no care-team sharing in this codebase — every threat below is single-user-account scoped.

## Session design and revocation

Sessions are stateless, signed JWT cookies (`lib/auth/session.ts`), not a server-side session store:

- Signed with HS256 using `AUTH_SECRET` (`jose`'s `SignJWT`); the app refuses to start signing if `AUTH_SECRET` is missing or under 32 characters (`secret()` throws).
- Payload is minimal: `{ sub: userId, tokenVersion, iss: 'dialog', iat, exp }` — no email, name, or health data ever goes in the cookie.
- Cookie flags: `httpOnly`, `sameSite: 'lax'`, `secure` in production, 30-day `maxAge`, scoped to `/`.
- **Revocation**: `User.tokenVersion` (an integer, default 0) is embedded in every signed token. `getCurrentUser()` (`lib/auth/current-user.ts`) rejects a valid, unexpired token if `user.tokenVersion !== session.tokenVersion` — bumping the database column invalidates every outstanding cookie for that user instantly ("sign out everywhere"), without needing a revocation list or server-side session store. That immediacy is the reason every mutation which bumps the column is a route handler rather than a Server Action — see [Settings mutations that revoke the current session](#settings-mutations-that-revoke-the-current-session).
- `middleware.ts` does a fast edge-level check (redirect unauthenticated visitors away from `/app/*`) but is explicitly documented as _not_ the authorization boundary — every server component and Server Action independently calls `requireUser()`/`requireOnboardedUser()` and re-verifies the session.

## Password handling

`lib/auth/password.ts`:

- Hashing: bcrypt (`bcryptjs`) at cost factor 12.
- Policy favours length over composition rules: minimum 10 characters, maximum 200, and rejection of a small explicit list of breach-list-dominant passwords (`password123`, `12345678`, etc.) — deliberately not a symbol-composition requirement, per the file's comment ("length beats composition rules... rather than forcing symbol soup that users write on a sticky note").
- Sign-in (`lib/actions/auth.ts`) always runs `bcrypt.compare()` even when no account matches the email, against a fixed dummy hash — so a nonexistent account and a wrong password take a similar amount of time, mitigating email enumeration via timing.
- Sign-up **does** return a specific "an account with that email already exists" error rather than a generic message — a deliberate, documented tradeoff (`lib/actions/auth.ts`'s comment: an attacker can already learn this by attempting to sign up, so vagueness here mostly punishes real users who forgot they have an account).

## Password recovery

`lib/auth/password-reset.ts`, `lib/auth/password-reset-token.ts`,
`app/api/auth/password/reset/route.ts`, and `lib/email/` implement recovery for
accounts that already have a password. Google-only accounts retain Google
sign-in; recovery never sets their first password.

- `/forgot-password` returns the same confirmation for existing, unknown, and
  Google-only accounts. Email throttling and delivery failures also remain
  neutral. Invalid email syntax, IP throttling, and deployment configuration
  failures can be reported without consulting account existence. Unlike the
  existing sign-up/sign-in responses, recovery must not become a mailbox-abuse
  oracle. Audits, token issuance and delivery run in Next's `after()` callback
  for every account state, so the neutral response never waits for those
  account-dependent operations. Configuration checks and the initial indexed
  database lookup remain on the response path; cache/load effects in that lookup
  can still vary, so this is not a claim of cryptographic constant-time lookup.
  `after()` keeps the task in Next's request lifecycle; a floating promise could
  be killed as soon as a serverless response returns and silently lose mail.
- Tokens contain 32 cryptographically random bytes encoded as base64url; only
  their SHA-256 digest is persisted. They expire after 30 minutes. Issuance and
  consumption lock the same account row. Issuance invalidates previous unused
  links; consumption conditionally sets `usedAt`, changes the bcrypt password,
  and increments `tokenVersion` in one transaction. Concurrent submissions can
  succeed only once. No account id submitted by the browser selects the owner.
- The emailed URL enters through `GET /api/auth/password/reset?token=…`, which
  verifies the raw token and redirects to `/reset-password?proof=…`. Next embeds
  page URLs in RSC responses, so this redirect-only entry is necessary to keep
  raw tokens out of response bodies. The form carries a signed, owner-bound
  proof with a separate `password-reset` audience and the original expiry.
  Possession of a database digest alone cannot forge that proof. A GET never
  consumes a link, so ordinary email link scanners do not invalidate it.
- The server component revalidates the proof against the database; the native
  form POST validates it again. POST requires an explicit same-origin `Origin`
  header, applies the existing password policy, and returns a 303 outcome.
  Success clears this browser's session, revokes every old session, and sends
  the user to sign in; it does not auto-sign-in. Recovery remains reachable
  with revoked or unverifiable cookies and fails closed without `AUTH_SECRET`.
- Requests are limited to 10 per IP per 15 minutes and 3 per normalized-email
  HMAC-SHA-256 key per hour; submissions have a separate 10/IP/15-minute budget.
  Keys contain no raw email. These inherit the existing limiter's per-process
  limitation and trusted-proxy/IP-header assumptions. Email keys use a separate,
  domain-separated HMAC function keyed by `AUTH_SECRET`, so a leaked bucket store
  cannot be tested against an email wordlist without that secret. A full process
  compromise that also exposes the secret defeats this protection. Persisted
  reset-token hashes remain bare SHA-256 and existing tokens stay compatible.
- Audits use `auth.password_reset_requested`, `auth.password_reset_completed`,
  and `auth.password_reset_failed`, with no token, proof, password or email in
  detail. Raw tokens never enter Prisma arguments or error messages. Console
  mail prints URLs only under the explicit development guard; tests capture
  mail in process memory. Production console delivery refuses to run. Relay
  failures emit a fixed operator signal, never a request/response body.
- Reset credentials are not health data: they survive `deleteAllRecordsAction`.
  The existing `PasswordResetToken.user` cascade deletes them with the account.

**Residual trust:** the product has no email-verification concept or
`emailVerified` field. Recovery proves present mailbox control, not that the
address originally belonged to the person who entered health data. A mistyped,
reassigned, shared, or compromised mailbox can expose the corresponding
password account. The mail relay and mailbox providers can read bearer links
and must be trusted. Proofs are also bearer credentials: exclude recovery URL
queries from proxy/APM/access logs, disable mail click tracking, and never
collect recovery pages in analytics. Redirects use `no-referrer`, and so does
the forgot-password page, whose URL carries nothing sensitive. The
**reset** page deliberately uses `strict-origin` instead: Chrome derives a form
submission's `Origin` header from the document's referrer policy, so
`no-referrer` there made the browser send `Origin: null` and the reset route —
which requires `Origin` — rejected every legitimate submission as
`cross_origin`. `strict-origin` sends the bare origin and never the path, so the
proof in that page's query string still never leaves in a `Referer`. The e2e
suite covers this end to end; a browser is the only place the interaction is
visible.
redirects are `no-store` and pages are dynamic. Browser/mail history remains a
residual exposure. Delivery runs after the response within Next's request
lifecycle, with no durable outbox or automatic retry; a transport failure
invalidates that token and the user must request a new link after the limit
permits it. Superseded/expired token rows are retained
until account deletion; deployments may add a retention job later.

## Federated sign-in (Google)

`lib/auth/oauth/*` and `app/api/auth/google/callback/route.ts`. Entirely optional — `googleConfig()` returns `null` and the app runs unaffected when `GOOGLE_CLIENT_ID`/`GOOGLE_CLIENT_SECRET` are unset (`.env.example`).

- **Identity is the Google `sub`, never the email.** `AuthIdentity` rows key on `(provider, providerAccountId)`, where `providerAccountId` is the token's stable `sub` claim. A returning user is recognised by that subject even if they have since renamed their Google address — email is only ever used to _find_ an account, not to prove ownership of one (`resolveGoogleSignIn` in `lib/auth/oauth/link.ts`).
- **Minimal scope, no token retention.** The authorize request asks for `openid email profile` only — never Gmail, Drive, or contacts. No health data leaves the deployment as part of this flow, and no Google access or refresh token is persisted anywhere: `verifyIdToken()` (`lib/auth/oauth/google.ts`) reads the ID token once, in memory, to extract `sub`/`email`/`email_verified`/`name`, and nothing from Google is written to the database beyond those fields.
- **PKCE + state + nonce in one signed cookie.** The attempt (`state`, PKCE `verifier`, OIDC `nonce`, `mode`, and post-sign-in `next`) is sealed into a single signed, HttpOnly cookie (`OAUTH_COOKIE`, `lib/auth/oauth/state.ts`) rather than several plaintext ones, so the callback validates the whole attempt — CSRF state match, PKCE code exchange, nonce replay — in one place. The cookie uses the authorization-code + PKCE (S256) flow and expires after `OAUTH_MAX_AGE_S` (10 minutes) and is single-use: the callback clears it on every response, success or failure.
- **ID token verification.** `verifyIdToken()` checks the signature against Google's live JWKS (`https://www.googleapis.com/oauth2/v3/certs`), and checks issuer, audience (the configured client id), and that the token's `nonce` matches the one minted for this attempt. Only a token that survives all of these is trusted; the callback treats every other input on the request — query parameters included — as attacker-controlled.
- **Deliberate non-linking-by-email policy.** A Google identity is never auto-linked to an existing DiaLog account by matching email, even when the email is verified. `resolveGoogleSignIn()` treats an email collision as `blocked` with `email_in_use` regardless of whether the existing account has a password or was itself created passwordlessly; `resolveGoogleLink()` only ever attaches a Google identity to the account already proven by an authenticated session. The only path to linking is: sign in with the password, then link from Settings. This is intentional, not an oversight — if a matching email were enough to merge accounts, whoever gained control of a person's Google account (a compromised, reused, or simply re-registered address) would silently inherit that person's entire health record. Google identity assertions alone never link accounts. Password recovery is a separate mailbox-control path for password accounts, with the unverified-address risks described above.

## Settings mutations that revoke the current session

Two failures were measured while building federated sign-in. In both, the mutation landed in the database and the server answered correctly — only the browser never applied the response, leaving a disabled button and no confirmation. Both are now fixed, and the rule that came out of them governs the whole settings surface.

**The root cause.** Revocation in DiaLog is a `User.tokenVersion` bump. The moment that column moves, _every_ cookie carrying the old value stops validating — including the one held by the tab performing the change. A Server Action can mint a replacement, but that replacement rides on exactly one response. Meanwhile the client router has other requests for the same document in flight — `<Link>` prefetches from the app shell, and the re-render Next performs after an action returns — and every one of them still carries the old cookie. Any that is served after the bump renders as signed-out (`middleware.ts` only checks the token's signature, so a stale cookie passes the edge and then fails `requireUser()` deep inside the render, which redirects to `/sign-in`). That poisons the router's cache entry for the page, and the action's result is discarded.

So the defect was never in the password logic. It was in the delivery: **a mutation that revokes the cookie its own request arrived with cannot report its outcome through the client router.** Measured on `changePasswordAction`, which bumps: 6 failures in 10. The same form setting a _first_ password, which does not bump: 0 in 10. `signOutEverywhereAction` had the identical shape and was simply never exercised repeatedly enough to be caught.

**The fix.** Every such mutation is now a route handler posted to by a plain `<form method="post">`:

| Mutation                 | Handler                                   |
| ------------------------ | ----------------------------------------- |
| Change / set password    | `app/api/auth/password/route.ts`          |
| Sign out everywhere else | `app/api/auth/sessions/revoke/route.ts`   |
| Disconnect Google        | `app/api/auth/google/disconnect/route.ts` |

A native form submission is a full-document navigation: the browser abandons the previous document and every request belonging to it, applies `Set-Cookie`, and follows the 303 with a GET already carrying the new token. There is no window in which a request holding the revoked cookie can still matter — the race is removed rather than narrowed. Outcomes travel as short codes on the query string (`?password=wrong_current`), which the settings page turns back into sentences and attaches to the field they belong to, so refusals keep the same accessible behaviour they had as returned field errors.

**What was deliberately not done.** The obvious alternative — letting a cookie one version behind stay valid for a grace period — was rejected. It would keep a _stolen_ cookie working for exactly the length of that grace, which is the opposite of what changing a password is for. Revocation stays immediate and absolute; only the delivery changed. `tests/e2e/account-security.spec.ts` asserts both halves: the confirmation appears on six consecutive changes, and a second signed-in browser is signed out by each one.

**Keeping it fixed.** Nothing about a Server Action prevents someone reintroducing the same shape later — the code would look entirely reasonable and fail one time in two, in a real browser only. Three layers guard against that:

- `tests/unit/auth/session-revocation.test.ts` fails the build if a `tokenVersion` write reappears anywhere in `lib/actions/*`, or if any of the three handlers loses its origin check or its 303, or if either settings form is bound back to a Server Action. It runs in the DB-free `npm test` suite.
- The shared plumbing in `lib/auth/route-form.ts` is the single implementation of the pattern (origin check, session read from the request, 303 helpers, cookie re-issue on the response), and its header explains the failure it exists to prevent.
- Outcome codes are typed: `PASSWORD_FEEDBACK` in `app/app/settings/page.tsx` is a `Record<PasswordOutcome, …>`, so adding a new outcome without giving it a message is a compile error rather than a silently blank confirmation.

### The other half: a revoked device could not reach the sign-in page

Fixing the delivery exposed a second defect in the same flow, found by the regression test asserting that other devices really are signed out. A device whose session had just been revoked hit `ERR_TOO_MANY_REDIRECTS` and could never recover on its own.

`middleware.ts` can only verify a token's _signature_ and expiry; revocation lives in `User.tokenVersion`, which needs a database read the edge cannot make. So a revoked cookie still parsed as a session there. The edge bounced `/sign-in` to `/app` on that basis, `/app` checked `tokenVersion` against the database and bounced back to `/sign-in`, and the browser gave up. The person was locked out of the one page that would have fixed it — the practical outcome of "sign out everywhere" on every other device.

The rule this produced: **an unverifiable token may be used to deny access to a protected route, never to deny access to the recovery page.** Middleware still guards `/app` (a stale cookie gets past it and is then rejected by `requireUser()`, the real authorization boundary), but the "already signed in, go to `/app`" convenience moved to the auth pages themselves, where `getCurrentUser()` can tell a live session from a revoked one. `tests/e2e/account-security.spec.ts` covers it from both directions: the revoked device lands on `/sign-in`, and the device that made the change stays signed in.

Route handlers do not get Next's automatic CSRF protection, so each checks the `Origin` header itself (`isSameOrigin()`); `sameSite: 'lax'` on the session cookie is the second layer, stopping a cross-site POST from carrying the session at all.

## Per-user authorization at the data layer

There is no row-level security at the Postgres level; authorization is enforced in application code, consistently:

- Every Prisma query that reads or writes a health record includes `userId: user.id` from the verified session — never a client-supplied id. `lib/db/health-records.ts` and every Server Action in `lib/actions/*` follow this pattern.
- `app/api/export/route.ts`'s own comment states the property directly: "There is no userId query parameter — the account is always the one attached to the session cookie, so this endpoint can never be pointed at someone else's data by editing the URL."
- `lib/services/export-service.ts`'s header comment: "Every query here is scoped by `userId` — there is no code path in this file that can read another account's data."
- Deletes/undo (`undoImport()` in `lib/services/import-service.ts`) look the target row up scoped by `userId` first (`findFirst({ where: { id, userId } })`) before deleting, so a batch id from another account simply matches nothing.
- This scoping is exercised by `tests/integration/db-health-records.test.ts` against a real Postgres instance, not just asserted in comments.

## Input validation

Every external input — form submissions, the export API's query params, imported files — is parsed before touching domain logic:

- Forms and Server Action payloads: Zod schemas in `lib/validation.ts` (`signUpSchema`, `glucoseEntrySchema`, etc.), each with user-facing error messages rather than developer-facing ones.
- AI structured output: Zod schemas in `lib/ai/schemas.ts`, applied to _model output_, not just user input — the model is treated as an untrusted input source too.
- Numeric health values: bounds-checked against physiological plausibility, not just type-checked (`isPlausibleGlucose()` in `lib/domain/units.ts`; systolic/diastolic/pulse ranges in `lib/validation.ts`).

## Upload handling

`lib/import/parse.ts` and `lib/services/import-service.ts`:

- Hard size ceiling: `MAX_FILE_BYTES = 100 MB`; a stricter `MAX_JSON_BYTES = 50 MB` for `JSON.parse`, which is O(n) memory on top of the string. `prepareImport()` rejects empty files and over-limit files before any parsing is attempted.
- Format detection and parsing (`parseFile()`) is defensive: unparseable files throw a caught, user-facing error rather than propagating a parser exception.
- **Two-stage commit**: `prepareImport()` parses and reports but writes nothing; `commitImport()` only runs after the user has seen a preview of what will happen. Nothing is corrected or silently altered — a row that can't be trusted becomes a visible `ImportIssue`, never a guessed value.
- Import writes are wrapped in `prisma.$transaction([...])` with `skipDuplicates: true` as a second, belt-and-braces dedupe layer beyond the pre-write dedupe-key check.

## Rate limiting — and its real limitation

`lib/auth/rate-limit.ts` implements fixed-window rate limiting with an **in-memory `Map`**:

```ts
export const RATE_LIMITS = {
  signIn: { limit: 10, windowMs: 15 * 60_000 },
  signUp: { limit: 5, windowMs: 60 * 60_000 },
  import: { limit: 20, windowMs: 60 * 60_000 },
  ai: { limit: 30, windowMs: 60 * 60_000 },
  write: { limit: 240, windowMs: 60 * 60_000 },
} as const;
```

applied on sign-in/sign-up by client IP (`clientKey()` in `lib/actions/auth.ts`, reading `x-forwarded-for`/`x-real-ip`), and per-user on AI calls (`lib/actions/assistant.ts`) and exports (`app/api/export/route.ts`).

**Honest limitation, stated in the module's own header comment**: "The in-memory store is per server instance, which is enough to blunt credential stuffing and import abuse on a single-region deployment." On Vercel (or any multi-instance/multi-region deployment), each serverless instance has its own independent `Map` — a client hitting different instances gets a fresh limit on each, so the effective limit is `per-instance limit × number of warm instances`, not the configured number. The interface (`rateLimit(key, limit, windowMs)`) is deliberately narrow so it can be swapped for a shared store (Redis / Vercel KV) without touching any call site — but that swap has not been made in this codebase. Treat the current limiter as a courtesy backstop against accidental abuse, not a hard guarantee under real distributed load.

## CSP and security headers (`next.config.ts`)

Applied to every response via `headers()`:

| Header                      | Value                                                                                                                                                                                                                                                                                                             | Purpose                                                                                                                                                                                                                                    |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `X-Content-Type-Options`    | `nosniff`                                                                                                                                                                                                                                                                                                         | Blocks MIME-sniffing.                                                                                                                                                                                                                      |
| `X-Frame-Options`           | `DENY`                                                                                                                                                                                                                                                                                                            | No embedding in a frame anywhere.                                                                                                                                                                                                          |
| `Referrer-Policy`           | `strict-origin-when-cross-origin`                                                                                                                                                                                                                                                                                 | Limits referrer leakage to other origins.                                                                                                                                                                                                  |
| `Permissions-Policy`        | `camera=(), microphone=(), geolocation=()`                                                                                                                                                                                                                                                                        | Denies device APIs DiaLog never uses.                                                                                                                                                                                                      |
| `Strict-Transport-Security` | `max-age=63072000; includeSubDomains; preload`                                                                                                                                                                                                                                                                    | Forces HTTPS for two years including subdomains.                                                                                                                                                                                           |
| `Content-Security-Policy`   | `default-src 'self'; script-src 'self' 'unsafe-inline'` (+`'unsafe-eval'` only in development, for Next's dev tooling)`; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self' data:; connect-src 'self'; form-action 'self'; frame-ancestors 'none'; base-uri 'self'; object-src 'none'` | No third-party script/style/connect origins anywhere; `'unsafe-inline'` on script/style is required because Next.js injects inline bootstrap scripts and the app uses inline chart CSS variables — noted directly in the config's comment. |

Additionally, every route under `/app/*` gets `X-Robots-Tag: noindex, nofollow, noarchive` so authenticated surfaces are never indexed.

## What is deliberately never logged

- **Request/response bodies for AI provider calls** — `lib/ai/providers/anthropic.ts`'s header comment states this directly: "Never logs request or response bodies (they contain health data) — only status codes and durations." The OpenAI provider follows the same pattern.
- **Health values in the audit log** — `AuditEvent` (`prisma/schema.prisma`) is explicitly comment-labeled "Security-relevant actions only. Never contains health values," and `audit()` (`lib/auth/audit.ts`) only ever receives `action`/`entity`/`entityId`/`detail` strings — callers pass things like a connector id or provider id, never a glucose value or meal description. Guardrail rejection notes (`lib/ai/guardrails.ts`) are similarly restricted to pattern labels, never the text that matched.
- **Prisma query logging is dev-only** — `lib/db/prisma.ts` enables `['warn', 'error']` logging only when `NODE_ENV === 'development'`; production logs only errors, not query parameters (which would include health values).
- **Audit writes never fail the request** — `audit()` wraps its own `prisma.auditEvent.create()` in try/catch so a logging failure can never break or reveal internals of the user-facing action.

## AI data-minimisation path

Covered fully in [AI_ARCHITECTURE.md](AI_ARCHITECTURE.md); the security-relevant summary:

1. The AI layer only ever receives an `EvidenceBundle` (aggregates + graded findings), never raw records — enforced both structurally (no import of `lib/analytics` types into `lib/ai`) and at runtime (`assertNoRawRecords()` in `lib/ai/pipeline.ts`, which throws if a raw-record-shaped value is detected).
2. `AI_PROVIDER` defaults to `local`, which makes zero network calls — health data leaves the deployment only if an operator explicitly configures `anthropic` or `openai` with an API key.
3. Even then, `redactForProvider()` (`lib/ai/redact.ts`) strips anything free-text-shaped from the bundle before it reaches an external provider, unless the user has explicitly set `Profile.externalAiConsentAt` — i.e. per-user, opt-in consent gates sending anything beyond aggregate numbers to a third party.

## If you are deploying this for real

- [ ] Generate a fresh, high-entropy `AUTH_SECRET` per environment (never reuse the value from `.env.example` or a dev `.env`) — see the Quick Start section of the README for the generation command.
- [ ] Put a shared rate-limit store (Redis, Vercel KV, or equivalent) behind `lib/auth/rate-limit.ts`'s interface before relying on rate limits under real multi-instance/multi-region load — the current in-memory limiter is per-instance only.
- [ ] Decide and document your `AI_PROVIDER` posture: `local` if you want a hard guarantee that health data never leaves your infrastructure; otherwise confirm your organisation's data-processing agreement with Anthropic/OpenAI covers the health data your users will consent to send.
- [ ] Put `DATABASE_URL` behind TLS (`sslmode=require` or your provider's default) and restrict network access to the database to your application's egress IPs where the provider supports it.
- [ ] Confirm your Postgres provider's backup policy and test a restore — nothing in this codebase implements backups.
- [ ] Set up log retention/monitoring for `AuditEvent` rows (`auth.sign_in_failed`, etc.) if you want to detect credential-stuffing patterns beyond what the rate limiter blunts.
- [ ] Review `next.config.ts`'s CSP if you add any third-party script, font, or analytics origin — the current policy allows none.
- [ ] Run `npx vitest run --config vitest.integration.config.ts` against a real (isolated, non-production) database as part of your deploy pipeline, not just the DB-free unit suite.
- [ ] Get a real security review / penetration test before handling real users' health data at scale — this document describes what is implemented, not an external audit's sign-off.
