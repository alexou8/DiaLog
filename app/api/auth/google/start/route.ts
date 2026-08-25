/**
 * Kick off Google sign-in (or connecting Google to the account you are already
 * signed in to, with `?mode=link`).
 */
import { NextResponse, type NextRequest } from 'next/server';
import {
  buildAuthorizeUrl,
  codeChallenge,
  googleConfig,
  randomToken,
} from '@/lib/auth/oauth/google';
import { OAUTH_COOKIE, OAUTH_MAX_AGE_S, safeNext, sealAttempt } from '@/lib/auth/oauth/state';
import { readSessionCookie } from '@/lib/auth/session';
import { RATE_LIMITS, pruneRateLimits, rateLimit } from '@/lib/auth/rate-limit';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

function bounce(request: NextRequest, path: string, code: string): NextResponse {
  const url = request.nextUrl.clone();
  url.pathname = path;
  url.search = `?error=${code}`;
  return NextResponse.redirect(url);
}

export async function GET(request: NextRequest): Promise<NextResponse> {
  const config = googleConfig();
  const linking = request.nextUrl.searchParams.get('mode') === 'link';

  if (!config) return bounce(request, linking ? '/app/settings' : '/sign-in', 'not_configured');

  // Linking is only meaningful for a signed-in person; without this an
  // unauthenticated visitor could burn the flow and land on Settings.
  const session = await readSessionCookie();
  if (linking && !session) return bounce(request, '/sign-in', 'invalid_state');

  pruneRateLimits();
  const ip =
    request.headers.get('x-forwarded-for')?.split(',')[0]?.trim() ??
    request.headers.get('x-real-ip') ??
    'unknown';
  const limit = rateLimit(
    `oauth:${ip}`,
    RATE_LIMITS.oauthStart.limit,
    RATE_LIMITS.oauthStart.windowMs,
  );
  if (!limit.ok) return bounce(request, linking ? '/app/settings' : '/sign-in', 'exchange_failed');

  const state = randomToken();
  const nonce = randomToken();
  const verifier = randomToken(48);

  const url = buildAuthorizeUrl(config, {
    state,
    nonce,
    challenge: await codeChallenge(verifier),
  });

  const response = NextResponse.redirect(url);
  response.cookies.set(
    OAUTH_COOKIE,
    await sealAttempt({
      state,
      nonce,
      verifier,
      mode: linking ? 'link' : 'signin',
      next: safeNext(request.nextUrl.searchParams.get('next'), linking ? '/app/settings' : '/app'),
    }),
    {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      // `lax` still sends the cookie on Google's top-level redirect back.
      sameSite: 'lax',
      path: '/',
      maxAge: OAUTH_MAX_AGE_S,
    },
  );
  return response;
}
