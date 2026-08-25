import { NextResponse, type NextRequest } from 'next/server';
import { SESSION_COOKIE, verifySession } from '@/lib/auth/session';

/**
 * Edge guard for authenticated surfaces. This is a fast rejection only —
 * every server component and API route independently re-checks the session
 * and scopes queries by user id.
 *
 * What the edge can and cannot decide
 * -----------------------------------
 * `verifySession()` proves a token was minted by this deployment and has not
 * expired. It cannot prove the token is still *current*: revocation lives in
 * `User.tokenVersion`, which needs a database read that middleware has no way
 * to make. So a cookie revoked by a password change or "sign out everywhere"
 * still looks perfectly valid here.
 *
 * That is fine for guarding `/app` — a stale cookie gets past the edge and is
 * then rejected by `requireUser()`, which is the real authorization boundary.
 * It is NOT fine as a reason to send someone *away* from the sign-in page.
 * Doing that used to trap every device that had just been signed out
 * remotely: the edge bounced /sign-in to /app because the token parsed, /app
 * checked `tokenVersion` and bounced back to /sign-in, and the browser gave up
 * with ERR_TOO_MANY_REDIRECTS. The person could not reach the one page that
 * would have fixed it.
 *
 * Redirecting an already-signed-in visitor from /sign-in to /app is a
 * convenience, and it belongs where the session can actually be verified —
 * the auth pages themselves, via `getCurrentUser()`. Nobody is ever locked out
 * of the recovery page by a check that cannot see revocation.
 */
const PROTECTED = ['/app'];

export async function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;
  const token = request.cookies.get(SESSION_COOKIE)?.value;
  const session = token ? await verifySession(token) : null;

  if (PROTECTED.some((p) => pathname === p || pathname.startsWith(`${p}/`)) && !session) {
    const url = request.nextUrl.clone();
    url.pathname = '/sign-in';
    url.searchParams.set('next', pathname);
    return NextResponse.redirect(url);
  }

  return NextResponse.next();
}

export const config = {
  matcher: ['/((?!_next/static|_next/image|favicon.ico|icons|manifest.webmanifest|sw.js).*)'],
};
