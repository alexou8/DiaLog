/**
 * Google's redirect back. Everything trusted here comes from the verified ID
 * token; the query string is treated as attacker-controlled throughout.
 */
import { NextResponse, type NextRequest } from 'next/server';
import { Prisma } from '@prisma/client';
import { prisma } from '@/lib/db/prisma';
import { audit } from '@/lib/auth/audit';
import { exchangeCode, googleConfig, verifyIdToken } from '@/lib/auth/oauth/google';
import { resolveGoogleLink, resolveGoogleSignIn, type OAuthErrorCode } from '@/lib/auth/oauth/link';
import { OAUTH_COOKIE, openAttempt, type OAuthAttempt } from '@/lib/auth/oauth/state';
import { readSessionCookie, signSession, SESSION_COOKIE } from '@/lib/auth/session';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const PROVIDER = 'google';

function redirectTo(request: NextRequest, path: string, params?: Record<string, string>) {
  const url = request.nextUrl.clone();
  url.pathname = path;
  url.search = params ? `?${new URLSearchParams(params).toString()}` : '';
  const response = NextResponse.redirect(url);
  // The attempt is single-use whatever the outcome.
  response.cookies.set(OAUTH_COOKIE, '', { httpOnly: true, path: '/', maxAge: 0 });
  return response;
}

function fail(request: NextRequest, attempt: OAuthAttempt | null, code: OAuthErrorCode) {
  const linking = attempt?.mode === 'link';
  return redirectTo(request, linking ? '/app/settings' : '/sign-in', { error: code });
}

async function withSession(
  response: NextResponse,
  user: { id: string; tokenVersion: number },
): Promise<NextResponse> {
  response.cookies.set(
    SESSION_COOKIE,
    await signSession({ userId: user.id, tokenVersion: user.tokenVersion }),
    {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'lax',
      path: '/',
      maxAge: 60 * 60 * 24 * 30,
    },
  );
  return response;
}

export async function GET(request: NextRequest): Promise<NextResponse> {
  const attempt = await openAttempt(request.cookies.get(OAUTH_COOKIE)?.value);
  const params = request.nextUrl.searchParams;

  const config = googleConfig();
  if (!config) return fail(request, attempt, 'not_configured');

  if (params.get('error')) {
    // `access_denied` is the ordinary "I changed my mind" case, not a fault.
    return fail(request, attempt, 'access_denied');
  }

  const code = params.get('code');
  // Constant work either way is pointless here; a mismatched state is simply
  // not a sign-in attempt this browser started.
  if (!attempt || !code || params.get('state') !== attempt.state) {
    return fail(request, attempt, 'invalid_state');
  }

  const tokens = await exchangeCode(config, code, attempt.verifier);
  if (!tokens) return fail(request, attempt, 'exchange_failed');

  const identity = await verifyIdToken(config, tokens.idToken, attempt.nonce);
  if (!identity) return fail(request, attempt, 'exchange_failed');

  const existingIdentity = await prisma.authIdentity.findUnique({
    where: {
      provider_providerAccountId: { provider: PROVIDER, providerAccountId: identity.subject },
    },
    select: { id: true, userId: true },
  });

  // ------------------------------------------------------ connecting Google

  if (attempt.mode === 'link') {
    const session = await readSessionCookie();
    if (!session) return fail(request, attempt, 'invalid_state');

    const current = await prisma.user.findUnique({
      where: { id: session.userId },
      select: {
        id: true,
        tokenVersion: true,
        identities: { where: { provider: PROVIDER }, select: { id: true } },
      },
    });
    if (!current || current.tokenVersion !== session.tokenVersion) {
      return fail(request, attempt, 'invalid_state');
    }

    const outcome = resolveGoogleLink(identity, {
      identityUserId: existingIdentity?.userId ?? null,
      currentUserId: current.id,
      currentUserHasGoogle: current.identities.length > 0,
    });

    if (outcome.kind === 'blocked') return fail(request, attempt, outcome.code);

    if (outcome.kind === 'link') {
      await prisma.authIdentity.create({
        data: {
          userId: current.id,
          provider: PROVIDER,
          providerAccountId: identity.subject,
          email: identity.email,
        },
      });
      await audit({ userId: current.id, action: 'auth.google_linked' });
    }

    return redirectTo(request, '/app/settings', {
      linked: outcome.kind === 'link' ? 'google' : 'google_already',
    });
  }

  // -------------------------------------------------------------- signing in

  const emailOwner = await prisma.user.findUnique({
    where: { email: identity.email },
    select: { id: true, passwordHash: true },
  });

  const outcome = resolveGoogleSignIn(identity, {
    identityUserId: existingIdentity?.userId ?? null,
    userWithEmail: emailOwner
      ? { id: emailOwner.id, hasPassword: emailOwner.passwordHash !== null }
      : null,
  });

  if (outcome.kind === 'blocked') {
    await audit({
      userId: emailOwner?.id ?? null,
      action: 'auth.google_sign_in_blocked',
      detail: outcome.code,
    });
    // Prefill the email so the person can go straight to typing their password.
    return redirectTo(request, '/sign-in', {
      error: outcome.code,
      ...(outcome.email ? { email: outcome.email } : {}),
    });
  }

  if (outcome.kind === 'create') {
    let created;
    try {
      created = await prisma.user.create({
        data: {
          email: identity.email,
          // No password: this account is reachable only through Google until
          // the person sets one from Settings.
          passwordHash: null,
          lastLoginAt: new Date(),
          profile: { create: { displayName: identity.name?.trim() || null } },
          identities: {
            create: {
              provider: PROVIDER,
              providerAccountId: identity.subject,
              email: identity.email,
              lastLoginAt: new Date(),
            },
          },
        },
        select: { id: true, tokenVersion: true },
      });
    } catch (error) {
      // Someone signed up with this email between our read and this write.
      if (error instanceof Prisma.PrismaClientKnownRequestError && error.code === 'P2002') {
        return redirectTo(request, '/sign-in', { error: 'email_in_use', email: identity.email });
      }
      throw error;
    }

    await audit({ userId: created.id, action: 'auth.sign_up', detail: 'google' });
    return withSession(redirectTo(request, '/app/onboarding'), created);
  }

  const user = await prisma.user.update({
    where: { id: outcome.userId },
    data: {
      lastLoginAt: new Date(),
      identities: {
        update: {
          where: { userId_provider: { userId: outcome.userId, provider: PROVIDER } },
          // Keep the displayed address current if they renamed it at Google.
          data: { email: identity.email, lastLoginAt: new Date() },
        },
      },
    },
    select: {
      id: true,
      tokenVersion: true,
      profile: { select: { onboardingCompletedAt: true } },
    },
  });

  await audit({ userId: user.id, action: 'auth.sign_in', detail: 'google' });
  const destination = user.profile?.onboardingCompletedAt ? attempt.next : '/app/onboarding';
  return withSession(redirectTo(request, destination), user);
}
