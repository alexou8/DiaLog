# Security

This document describes DiaLog's actual threat model and controls, citing the files that implement each one, and closes with a checklist for anyone deploying it for real.

## Threat model

DiaLog stores personal health data (glucose readings, meals, medications, symptoms, mood) for individual users. The controls below are built around three primary risks: **account takeover** (another party reading or writing a user's health records), **cross-user data leakage** (one user's data appearing in another user's view via a missing scope check), and **health data leaving the deployment** without the user's knowledge (via an external AI provider, logs, or a cache). There is no admin role, no multi-tenant organisation model, and no care-team sharing in this codebase — every threat below is single-user-account scoped.

## Session design and revocation

Sessions are stateless, signed JWT cookies (`lib/auth/session.ts`), not a server-side session store:

- Signed with HS256 using `AUTH_SECRET` (`jose`'s `SignJWT`); the app refuses to start signing if `AUTH_SECRET` is missing or under 32 characters (`secret()` throws).
- Payload is minimal: `{ sub: userId, tokenVersion, iss: 'dialog', iat, exp }` — no email, name, or health data ever goes in the cookie.
- Cookie flags: `httpOnly`, `sameSite: 'lax'`, `secure` in production, 30-day `maxAge`, scoped to `/`.
- **Revocation**: `User.tokenVersion` (an integer, default 0) is embedded in every signed token. `getCurrentUser()` (`lib/auth/current-user.ts`) rejects a valid, unexpired token if `user.tokenVersion !== session.tokenVersion` — bumping the database column invalidates every outstanding cookie for that user instantly ("sign out everywhere"), without needing a revocation list or server-side session store.
- `middleware.ts` does a fast edge-level check (redirect unauthenticated visitors away from `/app/*`) but is explicitly documented as _not_ the authorization boundary — every server component and Server Action independently calls `requireUser()`/`requireOnboardedUser()` and re-verifies the session.

## Password handling

`lib/auth/password.ts`:

- Hashing: bcrypt (`bcryptjs`) at cost factor 12.
- Policy favours length over composition rules: minimum 10 characters, maximum 200, and rejection of a small explicit list of breach-list-dominant passwords (`password123`, `12345678`, etc.) — deliberately not a symbol-composition requirement, per the file's comment ("length beats composition rules... rather than forcing symbol soup that users write on a sticky note").
- Sign-in (`lib/actions/auth.ts`) always runs `bcrypt.compare()` even when no account matches the email, against a fixed dummy hash — so a nonexistent account and a wrong password take a similar amount of time, mitigating email enumeration via timing.
- Sign-up **does** return a specific "an account with that email already exists" error rather than a generic message — a deliberate, documented tradeoff (`lib/actions/auth.ts`'s comment: an attacker can already learn this by attempting to sign up, so vagueness here mostly punishes real users who forgot they have an account).

## Federated sign-in (Google)

`lib/auth/oauth/*` and `app/api/auth/google/callback/route.ts`. Entirely optional — `googleConfig()` returns `null` and the app runs unaffected when `GOOGLE_CLIENT_ID`/`GOOGLE_CLIENT_SECRET` are unset (`.env.example`).

- **Identity is the Google `sub`, never the email.** `AuthIdentity` rows key on `(provider, providerAccountId)`, where `providerAccountId` is the token's stable `sub` claim. A returning user is recognised by that subject even if they have since renamed their Google address — email is only ever used to _find_ an account, not to prove ownership of one (`resolveGoogleSignIn` in `lib/auth/oauth/link.ts`).
- **Minimal scope, no token retention.** The authorize request asks for `openid email profile` only — never Gmail, Drive, or contacts. No health data leaves the deployment as part of this flow, and no Google access or refresh token is persisted anywhere: `verifyIdToken()` (`lib/auth/oauth/google.ts`) reads the ID token once, in memory, to extract `sub`/`email`/`email_verified`/`name`, and nothing from Google is written to the database beyond those fields.
- **PKCE + state + nonce in one signed cookie.** The attempt (`state`, PKCE `verifier`, OIDC `nonce`, `mode`, and post-sign-in `next`) is sealed into a single signed, HttpOnly cookie (`OAUTH_COOKIE`, `lib/auth/oauth/state.ts`) rather than several plaintext ones, so the callback validates the whole attempt — CSRF state match, PKCE code exchange, nonce replay — in one place. The cookie uses the authorization-code + PKCE (S256) flow and expires after `OAUTH_MAX_AGE_S` (10 minutes) and is single-use: the callback clears it on every response, success or failure.
- **ID token verification.** `verifyIdToken()` checks the signature against Google's live JWKS (`https://www.googleapis.com/oauth2/v3/certs`), and checks issuer, audience (the configured client id), and that the token's `nonce` matches the one minted for this attempt. Only a token that survives all of these is trusted; the callback treats every other input on the request — query parameters included — as attacker-controlled.
- **Deliberate non-linking-by-email policy.** A Google identity is never auto-linked to an existing DiaLog account by matching email, even when the email is verified. `resolveGoogleSignIn()` treats an email collision as `blocked` with `email_in_use` regardless of whether the existing account has a password or was itself created passwordlessly; `resolveGoogleLink()` only ever attaches a Google identity to the account already proven by an authenticated session. The only path to linking is: sign in with the password, then link from Settings. This is intentional, not an oversight — if a matching email were enough to merge accounts, whoever gained control of a person's Google account (a compromised, reused, or simply re-registered address) would silently inherit that person's entire health record. Requiring proof of the DiaLog password first means a Google account alone is never sufficient to reach someone else's data.

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
