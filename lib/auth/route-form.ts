/**
 * Plumbing for settings actions that change credentials or sessions.
 *
 * Why these are route handlers posted to by plain HTML forms, and not Server
 * Actions
 * -----------------------------------------------------------------------
 * An action that revokes the session cookie its own request arrived with
 * cannot reliably report its result through the client router.
 *
 * Revocation here is a `tokenVersion` bump (see lib/auth/session.ts): the
 * instant the column moves, *every* cookie carrying the old value stops
 * validating — including the one held by the tab making the change. The
 * handler can mint a replacement, but that replacement only rides on the one
 * response it is attached to. Meanwhile the client router has other requests
 * for the same document in flight — `<Link>` prefetches from the app shell,
 * the re-render Next performs after an action returns — and every one of them
 * still carries the old cookie. Any that is served after the bump renders as
 * signed-out: `requireUser()` redirects it to /sign-in, which poisons the
 * router's cache entry for the page and makes it discard the action result.
 * The mutation has landed, the server has answered correctly, and the person
 * is looking at a spinner that never resolves. Measured at 6 failures in 10 on
 * `changePasswordAction`, against 0 in 10 for the same form setting a *first*
 * password, which does not bump.
 *
 * The fix is not to retry, widen a grace period, or weaken revocation — a
 * grace period would keep a stolen cookie alive for exactly as long as it
 * lasted, which is the opposite of what changing a password is for. The fix is
 * to stop routing these responses through the client router at all. A plain
 * `<form method="post">` is a full-document navigation: the browser abandons
 * the old document and every request belonging to it, applies `Set-Cookie`,
 * and follows the 303 with a GET carrying the new cookie. There is no window
 * in which a request holding the revoked cookie can still matter, so the race
 * is not narrowed, it is removed.
 *
 * Outcomes therefore travel as short codes on the query string, which the
 * settings page turns back into sentences.
 *
 * Anything added here must keep both halves of that bargain:
 *   - the form is a plain `method="post"` form, never `useActionState`;
 *   - the handler answers with `back()`/`toSignIn()`, i.e. always a 303.
 *
 * `tests/unit/auth/session-revocation.test.ts` fails the build if a
 * `tokenVersion` write reappears in lib/actions/*.
 */
import { NextResponse, type NextRequest } from 'next/server';
import {
  SESSION_COOKIE,
  SESSION_COOKIE_OPTIONS,
  signSession,
  verifySession,
  type SessionPayload,
} from './session';

export const SETTINGS_PATH = '/app/settings';

/**
 * Server Actions get CSRF protection from Next; a route handler has to do its
 * own. Only a form served from this origin may reach these handlers.
 *
 * A missing `Origin` is allowed because some browsers omit it on same-origin
 * form posts; `sameSite: 'lax'` on the session cookie is what stops a
 * cross-site POST from carrying the session in the first place.
 */
export function isSameOrigin(request: NextRequest): boolean {
  const origin = request.headers.get('origin');
  return !origin || origin === request.nextUrl.origin;
}

/**
 * Read the session from the request itself, rather than from the ambient
 * `cookies()` store.
 *
 * These handlers are given a request and answer with a response; taking the
 * cookie from that request keeps them pure functions of their input, which is
 * what lets tests/integration/auth-credential-routes.test.ts call them directly
 * and assert on the `Set-Cookie` they return. Verification is unchanged — the
 * caller still has to re-check `tokenVersion` against the database, because a
 * signed token only proves the token was minted here, not that it is current.
 */
export async function sessionFromRequest(request: NextRequest): Promise<SessionPayload | null> {
  const token = request.cookies.get(SESSION_COOKIE)?.value;
  return token ? verifySession(token) : null;
}

export function crossOrigin(): NextResponse {
  return NextResponse.json({ error: 'cross_origin' }, { status: 403 });
}

/**
 * Redirect back to Settings carrying one outcome code.
 *
 * 303 so the browser follows with a GET whatever it posted, and so a reload of
 * the resulting page never re-submits the form.
 */
export function back(request: NextRequest, key: string, outcome: string): NextResponse {
  const url = request.nextUrl.clone();
  url.pathname = SETTINGS_PATH;
  url.search = `?${key}=${encodeURIComponent(outcome)}`;
  return NextResponse.redirect(url, 303);
}

/** Send an unauthenticated (or newly de-authenticated) request to sign in. */
export function toSignIn(request: NextRequest): NextResponse {
  const url = request.nextUrl.clone();
  url.pathname = '/sign-in';
  url.search = '';
  return NextResponse.redirect(url, 303);
}

/**
 * Attach a freshly minted session cookie to a response.
 *
 * Written onto the response rather than the ambient cookie store because these
 * handlers answer with a redirect: the `Set-Cookie` and the `Location` must
 * arrive together, so that the GET the browser makes next already carries the
 * new token.
 */
export async function issueSession(
  response: NextResponse,
  userId: string,
  tokenVersion: number,
): Promise<NextResponse> {
  const token = await signSession({ userId, tokenVersion });
  response.cookies.set(SESSION_COOKIE, token, SESSION_COOKIE_OPTIONS);
  return response;
}
