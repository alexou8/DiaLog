'use server';

import { headers } from 'next/headers';
import { redirect } from 'next/navigation';
import { prisma } from '@/lib/db/prisma';
import { audit } from '@/lib/auth/audit';
import {
  dummyPasswordHash,
  hashPassword,
  validatePassword,
  verifyPassword,
} from '@/lib/auth/password';
import {
  RATE_LIMITS,
  accountRateLimitKey,
  pruneRateLimits,
  rateLimit,
  resetRateLimit,
} from '@/lib/auth/rate-limit';
import {
  clearSessionCookie,
  isSessionSecretConfigured,
  readSessionCookie,
  setSessionCookie,
  signSession,
} from '@/lib/auth/session';
import { fieldErrors, signInSchema, signUpSchema } from '@/lib/validation';
import { safeNext } from '@/lib/auth/oauth/state';

export interface ActionState {
  ok: boolean;
  message?: string;
  errors?: Record<string, string>;
}

/**
 * A deployment with no usable `AUTH_SECRET` cannot sign a session cookie, so
 * `signSession()` throws. Checked up front in both credential actions because
 * the throw would otherwise escape the action uncaught and React would replace
 * the whole sign-in page with its generic error screen — which reads as "the
 * site is broken" rather than "this deployment is missing a setting", and hides
 * the one detail an operator needs. Nothing is created and no session is issued
 * in this state; only the message changes.
 */
const NOT_CONFIGURED: ActionState = {
  ok: false,
  message:
    'Sign-in is unavailable because this deployment is not fully configured. If you administer it, set AUTH_SECRET (32+ characters) in the environment and redeploy.',
};

/**
 * One message for both sign-in budgets (per IP and per account). Saying which
 * one tripped would tell a caller whether their address or the account was
 * the bottleneck, which is more than a legitimate user needs.
 */
function tooManySignIns(retryAfterSeconds: number): ActionState {
  return {
    ok: false,
    message: `Too many sign-in attempts. Please wait about ${Math.ceil(retryAfterSeconds / 60)} minutes and try again.`,
  };
}

/** Best-effort client identifier for rate limiting. */
async function clientKey(prefix: string): Promise<string> {
  const h = await headers();
  const ip = h.get('x-forwarded-for')?.split(',')[0]?.trim() ?? h.get('x-real-ip') ?? 'unknown';
  return `${prefix}:${ip}`;
}

export async function signUpAction(
  _prev: ActionState | null,
  formData: FormData,
): Promise<ActionState> {
  if (!isSessionSecretConfigured()) return NOT_CONFIGURED;

  pruneRateLimits();
  const limit = rateLimit(
    await clientKey('signup'),
    RATE_LIMITS.signUp.limit,
    RATE_LIMITS.signUp.windowMs,
  );
  if (!limit.ok) {
    return {
      ok: false,
      message: 'Too many accounts have been created from this connection. Please try again later.',
    };
  }

  const parsed = signUpSchema.safeParse(Object.fromEntries(formData));
  if (!parsed.success) return { ok: false, errors: fieldErrors(parsed.error) };

  const policy = validatePassword(parsed.data.password);
  if (!policy.ok) return { ok: false, errors: { password: policy.message } };

  const existing = await prisma.user.findUnique({
    where: { email: parsed.data.email },
    select: { id: true },
  });
  if (existing) {
    // Deliberately specific: an attacker can already learn this by trying to
    // sign up, and being vague here mostly punishes real users who forgot.
    return {
      ok: false,
      errors: { email: 'An account with that email already exists. Try signing in instead.' },
    };
  }

  const user = await prisma.user.create({
    data: {
      email: parsed.data.email,
      passwordHash: await hashPassword(parsed.data.password),
      profile: {
        create: {
          displayName: parsed.data.displayName?.trim() || null,
        },
      },
    },
  });

  await audit({ userId: user.id, action: 'auth.sign_up' });
  await setSessionCookie(await signSession({ userId: user.id, tokenVersion: user.tokenVersion }));
  redirect('/app/onboarding');
}

export async function signInAction(
  _prev: ActionState | null,
  formData: FormData,
): Promise<ActionState> {
  if (!isSessionSecretConfigured()) return NOT_CONFIGURED;

  pruneRateLimits();
  const key = await clientKey('signin');
  const limit = rateLimit(key, RATE_LIMITS.signIn.limit, RATE_LIMITS.signIn.windowMs);
  if (!limit.ok) return tooManySignIns(limit.retryAfterSeconds);

  const parsed = signInSchema.safeParse(Object.fromEntries(formData));
  if (!parsed.success) return { ok: false, errors: fieldErrors(parsed.error) };

  // The per-IP budget alone let a distributed attacker spread guesses at one
  // account across many addresses with no account-level brake. Applied before
  // the lookup, and identically for unknown addresses, so the limiter answers
  // the same whether or not the account exists (see accountRateLimitKey).
  const accountKey = accountRateLimitKey('signin-account', parsed.data.email);
  const accountLimit = rateLimit(
    accountKey,
    RATE_LIMITS.signInAccount.limit,
    RATE_LIMITS.signInAccount.windowMs,
  );
  if (!accountLimit.ok) return tooManySignIns(accountLimit.retryAfterSeconds);

  const user = await prisma.user.findUnique({
    where: { email: parsed.data.email },
    include: {
      profile: { select: { onboardingCompletedAt: true } },
      identities: { select: { provider: true } },
    },
  });

  // Signed up through Google and never set a password: say so plainly instead
  // of insisting the password is wrong, which would look like a lost account.
  if (user && user.passwordHash === null) {
    const provider = user.identities.some((i) => i.provider === 'google') ? 'Google' : null;
    return {
      ok: false,
      message: provider
        ? `This account signs in with ${provider}. Use the “Sign in with ${provider}” button below, then set a password from Settings if you would like one.`
        : 'This account does not have a password set. Please use the sign-in method you registered with.',
    };
  }

  // Always run a full comparison so that a missing account and a wrong password
  // take the same time. The fallback must be a real hash at the real cost:
  // bcryptjs short-circuits on a malformed one (see dummyPasswordHash).
  const hash = user?.passwordHash ?? (await dummyPasswordHash());
  const valid = await verifyPassword(parsed.data.password, hash);

  if (!user || !valid) {
    await audit({ userId: user?.id ?? null, action: 'auth.sign_in_failed' });
    return {
      ok: false,
      message: 'That email and password do not match an account. Please check both and try again.',
    };
  }

  // The account budget exists to brake password guessing, so only failures
  // should spend it. Charged up front (atomic with the check) and refunded
  // here: without the refund, 20 ordinary sign-ins in 15 minutes locked a
  // shared account like the demo out for every visitor.
  resetRateLimit(accountKey);
  await prisma.user.update({ where: { id: user.id }, data: { lastLoginAt: new Date() } });
  await audit({ userId: user.id, action: 'auth.sign_in' });
  await setSessionCookie(await signSession({ userId: user.id, tokenVersion: user.tokenVersion }));
  // Honour a same-origin `next` (deep links, PWA shortcuts) the same way the
  // Google callback does; safeNext keeps it on this origin.
  redirect(
    user.profile?.onboardingCompletedAt
      ? safeNext(formData.get('next')?.toString(), '/app')
      : '/app/onboarding',
  );
}

export async function signOutAction(): Promise<void> {
  const session = await readSessionCookie();
  await audit({ userId: session?.userId ?? null, action: 'auth.sign_out' });
  await clearSessionCookie();
  redirect('/');
}
