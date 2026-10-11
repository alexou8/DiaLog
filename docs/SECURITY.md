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
- Policy favours length over composition rules: minimum 10 characters, maximum **72 UTF-8 bytes** (`PASSWORD_MAX_BYTES`), and rejection of a small explicit list of breach-list-dominant passwords (`password123`, `12345678`, etc.) — deliberately not a symbol-composition requirement, per the file's comment ("length beats composition rules... rather than forcing symbol soup that users write on a sticky note").
- The 72-byte ceiling is bcrypt's own: it reads only the first 72 bytes of its input and ignores the rest, so the old 200-character maximum accepted characters that did nothing (two passwords sharing a 72-byte prefix were the same password). The cap is in bytes, not characters — an accented letter is 2 bytes and most emoji 4 — and the refusal says so. `tests/unit/auth/password-policy.test.ts`.
- Sign-in (`lib/actions/auth.ts`) always runs `bcrypt.compare()` even when no account matches the email, against `dummyPasswordHash()` — a real bcrypt hash, at the real cost (`PASSWORD_HASH_ROUNDS`), of a random password, generated once per process. The previous fallback was a hand-typed 65-character string; bcryptjs returns `false` immediately for any hash that is not 60 characters, so an unknown email answered in ~0 ms against ~250 ms for a known one — the comment promised equal timing and the code did not deliver it. Sign-up already discloses whether an address is registered (below), so the practical impact was low, but the guarantee is now real and unit-tested.
- Sign-in is rate limited per IP **and per account** — see "Rate limiting" below.
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

**A completed reset removes every linked sign-in identity.** Sign-up does not
verify the email address, which allowed an account pre-hijack: someone registers
`victim@example.com` first, links their own Google account from Settings, and
waits. When the real owner later recovers the account by email, the reset used
to change the password and bump `tokenVersion` but leave the squatter's
`AuthIdentity` row in place, so they kept signing in through Google. Proving
control of the mailbox is the strongest claim the account has, so the reset
transaction (`app/api/auth/password/reset/route.ts`) now also deletes every
`AuthIdentity` for the user, audits `auth.google_unlinked` with detail
`password_reset`, and adds `unlinked=google` to the redirect to `/sign-in` so the
page can tell the person (the sign-in page does not render that yet — UI
follow-up). Sessions are still revoked as before. The owner can reconnect Google
from Settings. A failed or refused reset unlinks nothing.
`tests/integration/password-reset.test.ts`. The root cause — no email
verification before linking — is not fixed; verifying email at sign-up (or
before a provider can be linked) remains the better long-term control.

Recovery also skips the shared demo account (see "The shared demo account"):
the request still answers with the neutral confirmation, but no link is issued,
and a link minted before that rule is refused as `invalid` at submission.

## Federated sign-in (Google)

`lib/auth/oauth/*` and `app/api/auth/google/callback/route.ts`. Entirely optional — `googleConfig()` returns `null` and the app runs unaffected when `GOOGLE_CLIENT_ID`/`GOOGLE_CLIENT_SECRET` are unset (`.env.example`).

- **Identity is the Google `sub`, never the email.** `AuthIdentity` rows key on `(provider, providerAccountId)`, where `providerAccountId` is the token's stable `sub` claim. A returning user is recognised by that subject even if they have since renamed their Google address — email is only ever used to _find_ an account, not to prove ownership of one (`resolveGoogleSignIn` in `lib/auth/oauth/link.ts`).
- **Minimal scope, no token retention.** The authorize request asks for `openid email profile` only — never Gmail, Drive, or contacts. No health data leaves the deployment as part of this flow, and no Google access or refresh token is persisted anywhere: `verifyIdToken()` (`lib/auth/oauth/google.ts`) reads the ID token once, in memory, to extract `sub`/`email`/`email_verified`/`name`, and nothing from Google is written to the database beyond those fields.
- **PKCE + state + nonce in one signed cookie.** The attempt (`state`, PKCE `verifier`, OIDC `nonce`, `mode`, and post-sign-in `next`) is sealed into a single signed, HttpOnly cookie (`OAUTH_COOKIE`, `lib/auth/oauth/state.ts`) rather than several plaintext ones, so the callback validates the whole attempt — CSRF state match, PKCE code exchange, nonce replay — in one place. The cookie uses the authorization-code + PKCE (S256) flow and expires after `OAUTH_MAX_AGE_S` (10 minutes) and is single-use: the callback clears it on every response, success or failure.
- **ID token verification.** `verifyIdToken()` checks the signature against Google's live JWKS (`https://www.googleapis.com/oauth2/v3/certs`), and checks issuer, audience (the configured client id), and that the token's `nonce` matches the one minted for this attempt. Only a token that survives all of these is trusted; the callback treats every other input on the request — query parameters included — as attacker-controlled.
- **Deliberate non-linking-by-email policy.** A Google identity is never auto-linked to an existing DiaLog account by matching email, even when the email is verified. `resolveGoogleSignIn()` treats an email collision as `blocked` with `email_in_use` regardless of whether the existing account has a password or was itself created passwordlessly; `resolveGoogleLink()` only ever attaches a Google identity to the account already proven by an authenticated session. The only path to linking is: sign in with the password, then link from Settings. This is intentional, not an oversight — if a matching email were enough to merge accounts, whoever gained control of a person's Google account (a compromised, reused, or simply re-registered address) would silently inherit that person's entire health record. Google identity assertions alone never link accounts. Password recovery is a separate mailbox-control path for password accounts, with the unverified-address risks described above — and a completed recovery removes any linked Google identity, so a link made by someone who registered the address first does not survive the real owner recovering it.
- **The demo account cannot link or unlink.** `start/route.ts` refuses `mode=link` for it, and the callback refuses again before writing (`demo_read_only`); only the callback check is binding, because the session at callback time need not be the one that started the flow.

## The shared demo account

`prisma/seed.ts` creates `demo@dialog.health` with a password published in the
README, so anyone can sign in as it. Without limits, one visitor could change
its password or "sign out everywhere" (locking every later visitor out until a
re-seed), attach their own Google account as a backdoor, or delete the account
or all its records. `lib/auth/demo.ts` names the account (`DEMO_EMAIL`, matched
case-insensitively — there is no `isDemo` column), and these paths refuse it
before doing anything:

| Path                                                                                            | Refusal                                                                        |
| ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `POST /api/auth/password`, `POST /api/auth/sessions/revoke`, `POST /api/auth/google/disconnect` | 303 to `/app/settings?error=demo_read_only`; nothing written, no cookie issued |
| `GET /api/auth/google/start?mode=link`, and the link branch of the callback                     | same `?error=demo_read_only`; no `AuthIdentity` created                        |
| `deleteAccountAction`, `deleteAllRecordsAction`                                                 | `{ ok: false, message: DEMO_READ_ONLY_MESSAGE }`                               |
| Password recovery request / submission                                                          | neutral confirmation, no link issued / `invalid`                               |
| `updatePreferencesAction` granting external-AI consent                                          | `{ ok: false, message: DEMO_READ_ONLY_MESSAGE }`; other preferences still save |

The message names only the demo account itself. `?error=demo_read_only` is
rendered by Settings through the same `oauthMessage()` lookup as Google errors;
note that Settings only shows that notice inside the Connected accounts card,
which is hidden when Google is not configured — a UI follow-up, not a security
gap (the refusal itself does not depend on it). Tests:
`tests/integration/demo-account.test.ts`, `tests/unit/auth/demo.test.ts` (which
also fails if `DEMO_EMAIL` drifts from `prisma/seed.ts`).

Consent is personal but the demo is shared, so one visitor's consent would
have sent every later visitor's questions and quick-log text to an external
provider. Besides refusing the grant, `lib/actions/assistant.ts` treats the
demo as never consenting, even if its row carries a consent timestamp from
before the lock. Withdrawing consent stays possible.

Deliberately still allowed: deleting single records and undoing an import.
They are part of trying the app, and the scheduled re-seed restores the data.

Not covered here (other workstreams): the demo can still run imports and AI
calls, which the October 2026 review also flagged as an anonymous springboard.

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
- Pagination cursors are client-supplied ids too. `pageGlucose()` used Prisma's `cursor: { id }`, which resolves the cursor row by id alone — outside the `userId` filter — so another account's reading id positioned this user's History page at that reading's timestamp (an ordering oracle). It now looks the cursor up with `findFirst({ where: { id, userId } })` and builds the keyset condition itself; an unknown or foreign cursor restarts from the first page. **Known remaining instance:** the non-glucose History tabs in `app/app/history/page.tsx` still pass `cursor: { id }` directly to Prisma (outside this change's scope); the same fix applies there.
- `deleteOwnedRecord()` deletes with `deleteMany({ where: { id, userId } })` for every type in `DELETABLE_RECORD_TYPES`, which now includes hydration and symptom entries (previously creatable but not individually deletable). The Server Action that calls it still narrows with `isRecordType()` until it switches to `isDeletableRecordType()`.
- This scoping is exercised by `tests/integration/db-health-records.test.ts` against a real Postgres instance, not just asserted in comments.

## Erasure ("Delete all my records")

`deleteAllRecordsAction` (`lib/actions/preferences.ts`) delegates to
`deleteAllHealthData(userId)` in `lib/db/health-records.ts`, one transaction
over: the eleven record tables (FoodItem cascades from Meal), `AIConversation`
(→ `AIMessage`), `ImportBatch` (→ `ImportIssue.rawRow`), **`Insight`** (its
summary, detail and evidence are statements about the user's readings; nothing
writes it today, which is exactly when such a gap is missed) and **`Device`**
(label, vendor and model say which meter or CGM the person uses). The function
lives outside the `'use server'` file deliberately: every exported async
function of such a module is a client-callable endpoint, and a `(userId) => …`
export there would let any client erase any account.
`tests/integration/data-lifecycle.test.ts` now drives the real Server Action
(with only `next/headers` and `next/cache` stood in for) and checks every model
is empty for the target user and untouched for a bystander. Account deletion
still relies on `onDelete: Cascade` from `User`, with `AuditEvent` the
deliberate `SetNull` exception.

## Input validation

Every external input — form submissions, the export API's query params, imported files — is parsed before touching domain logic:

- Forms and Server Action payloads: Zod schemas in `lib/validation.ts` (`signUpSchema`, `glucoseEntrySchema`, etc.), each with user-facing error messages rather than developer-facing ones.
- AI structured output: Zod schemas in `lib/ai/schemas.ts`, applied to _model output_, not just user input — the model is treated as an untrusted input source too.
- Numeric health values: bounds-checked against physiological plausibility, not just type-checked (`isPlausibleGlucose()` in `lib/domain/units.ts`; systolic/diastolic/pulse ranges in `lib/validation.ts`).
- Values later handed to `Intl`, which throws `RangeError` on bad input and turned a bad value into a 500 or a permanently broken profile:
  - Date-times (`localDateTimeSchema`) must name a real instant — `isRealLocalDateTime()` builds the date in UTC and compares every component back, because `Date` silently rolls `2026-02-30` forward rather than failing. `2025-13-45T99:99` used to pass the regex.
  - Time zones (`isSupportedTimeZone()`): allowed if in `Intl.supportedValuesOf('timeZone')` plus `UTC`, **or** IANA-shaped and accepted by `new Intl.DateTimeFormat`. The fallback exists because the canonical spelling differs between ICU builds — Node 24 lists `Asia/Calcutta`, current browsers report `Asia/Kolkata`, which is what onboarding submits — and anything `Intl` accepts cannot cause the RangeError this guards against. Offsets such as `+05:00` are refused.
  - Locales (`isValidLocale()`): `Intl.getCanonicalLocales` in try/catch.
  - `tests/unit/domain/validation.test.ts`, which also asserts every zone and locale the UI offers is accepted.

## Upload handling

`lib/import/parse.ts` and `lib/services/import-service.ts`:

- Hard size ceiling: `MAX_IMPORT_BYTES` (20 MB) in `lib/import/limits.ts`, which `MAX_FILE_BYTES`/`MAX_JSON_BYTES` reuse. `prepareImport()` rejects empty files and over-limit files before any parsing is attempted.
- Uploads arrive through a Server Action, whose request body Next caps at 1 MB by default — so files between 1 MB and the import ceiling used to fail with a bare framework error. `next.config.ts` sets `experimental.serverActions.bodySizeLimit: '21mb'` — 1 MB above the ceiling because the multipart envelope counts towards the limit (a literal, not an import, so the config stays free of app code; change both together). This limit applies to every Server Action, not only imports. One caveat it cannot fix: a hosting platform's own request-body cap wins if lower (**inference from Vercel's published limits, not tested here:** Vercel Functions document 4.5 MB).
- Format detection and parsing (`parseFile()`) is defensive: unparseable files throw a caught, user-facing error rather than propagating a parser exception.
- **Two-stage commit**: `prepareImport()` parses and reports but writes nothing; `commitImport()` only runs after the user has seen a preview of what will happen. Nothing is corrected or silently altered — a row that can't be trusted becomes a visible `ImportIssue`, never a guessed value.
- Import writes are wrapped in `prisma.$transaction([...])` with `skipDuplicates: true` as a second, belt-and-braces dedupe layer beyond the pre-write dedupe-key check.

## Rate limiting — and its real limitation

`lib/auth/rate-limit.ts` implements fixed-window rate limiting with an **in-memory `Map`**:

| Budget                | Limit / window | Keyed by                                           |
| --------------------- | -------------- | -------------------------------------------------- |
| `signIn`              | 10 / 15 min    | client IP                                          |
| `signInAccount`       | 20 / 15 min    | the email typed, HMAC'd (`accountRateLimitKey`)    |
| `oauthStart`          | 30 / 15 min    | client IP                                          |
| `signUp`              | 5 / hour       | client IP                                          |
| `passwordResetIp`     | 10 / 15 min    | client IP                                          |
| `passwordResetEmail`  | 3 / hour       | the email typed, HMAC'd (`keyedRecoveryEmailHash`) |
| `passwordResetSubmit` | 10 / 15 min    | client IP                                          |
| `import`, `ai`        | 20, 30 / hour  | user                                               |
| `write`               | 240 / hour     | user                                               |

Client IP comes from the first `x-forwarded-for` hop or `x-real-ip`. That is
only trustworthy where the platform overwrites those headers (Vercel does); on
another host a client can set them and pick its own IP bucket.

**Per-account sign-in limiting.** The IP budget stops one client trying many
accounts; it did nothing about many clients (or one rotating addresses) trying
one account. `signInAction` now also spends an account bucket keyed by an HMAC
of the lowercased, trimmed email — applied before the user lookup and
identically for unknown addresses, so the limiter is not itself an
account-existence oracle, and the bucket map never holds a raw address. Both
budgets produce the same message. The account budget is deliberately looser
than the IP one because anyone can spend it on a victim's behalf: a deliberate
lockout lasts at most 15 minutes and leaves recovery and Google sign-in
working. Password-reset _requests_ already had a per-account bucket
(`passwordResetEmail`); reset _submissions_ need a server-signed proof, so a
per-IP budget is enough there. `tests/unit/auth/rate-limit.test.ts` drives the
real `signInAction` across many IPs.

**Honest limitation, stated in the module's own header comment**: the store is an in-memory `Map` per server instance. On Vercel (or any multi-instance/multi-region deployment), each serverless instance has its own independent `Map` — a client hitting different instances gets a fresh limit on each, so the effective limit is `per-instance limit × number of warm instances`, not the configured number. The interface (`rateLimit(key, limit, windowMs)`) is deliberately narrow so it can be swapped for a shared store (Redis / Vercel KV) without touching any call site — but that swap has not been made in this codebase (a Postgres-backed limiter — a table keyed by `(key, window)` with `INSERT … ON CONFLICT DO UPDATE … RETURNING count` — is the deferred plan). Treat the current limiter as a courtesy backstop against accidental abuse, not a hard guarantee under real distributed load. The per-account sign-in bucket has the same limitation: a distributed guesser spread across N warm instances gets up to N × 20 guesses per account per 15 minutes.

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

**Known gap: `'unsafe-inline'` in `script-src`.** It means the CSP does not stop an injected inline `<script>`, so most of its XSS value is forfeit and React's output escaping is doing that job alone. The fix is a per-request nonce set in middleware plus `'strict-dynamic'` (supported by Next 15), which also forces every page to render dynamically; it is deferred, and `next.config.ts` says so next to the directive.

## Data export

`app/api/export/route.ts` and `lib/services/export-service.ts`:

- **Complete, never silently truncated.** Exports used to stop at 20,000 rows per record type — about ten weeks of CGM readings at one per five minutes — so "download all my data" returned a fraction of a long history. Records are now read in keyset pages of `EXPORT_PAGE_SIZE` (2,000) ordered by `(takenAt, id)` and streamed to the client as they arrive, so memory stays at one page and there is no ceiling. If a page fails mid-way the response stream is errored and the download fails visibly, rather than ending early with a valid-looking file; only a fixed line is logged, never the error (which could carry query arguments). The export is not a point-in-time snapshot: a record written during an export may or may not be included. JSON `schemaVersion` is now 2 (`rowLimit: null`; `meta.*.truncated` always `false`). The old `EXPORT_ROW_LIMIT` constant has been removed and the Settings copy no longer mentions a cap.
- **CSV formula injection is neutralised.** A text cell starting with `=`, `+`, `-`, `@`, tab or CR — a note or meal description, often carried in from a third-party import — runs as a formula when the CSV is opened in Excel, Sheets or LibreOffice. Such cells are prefixed with `'`. Only text columns are treated; numeric columns are written as numbers, so a negative value is not turned into text. The JSON export is left byte-for-byte, since it is not a spreadsheet format.
- Tests: `tests/integration/export-complete.test.ts` (many page boundaries with ties on `takenAt`, one real page boundary at the production page size, neutralisation per column), `tests/unit/domain/validation-export-csv.test.ts`.

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
- [ ] Decide whether to seed the public demo account at all. Its credentials and deletion are locked (see "The shared demo account"), but it remains a no-signup way into imports and the assistant.
- [ ] Check your host's request-body limit against the 20 MB import ceiling; a lower platform cap (inference: 4.5 MB on Vercel Functions) silently wins.
- [ ] Decide and document your `AI_PROVIDER` posture: `local` if you want a hard guarantee that health data never leaves your infrastructure; otherwise confirm your organisation's data-processing agreement with Anthropic/OpenAI covers the health data your users will consent to send.
- [ ] Put `DATABASE_URL` behind TLS (`sslmode=require` or your provider's default) and restrict network access to the database to your application's egress IPs where the provider supports it.
- [ ] Confirm your Postgres provider's backup policy and test a restore — nothing in this codebase implements backups.
- [ ] Set up log retention/monitoring for `AuditEvent` rows (`auth.sign_in_failed`, etc.) if you want to detect credential-stuffing patterns beyond what the rate limiter blunts.
- [ ] Review `next.config.ts`'s CSP if you add any third-party script, font, or analytics origin — the current policy allows none.
- [ ] Run `npx vitest run --config vitest.integration.config.ts` against a real (isolated, non-production) database as part of your deploy pipeline, not just the DB-free unit suite.
- [ ] Get a real security review / penetration test before handling real users' health data at scale — this document describes what is implemented, not an external audit's sign-off.
