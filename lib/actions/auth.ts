'use server';

import { headers } from 'next/headers';
import { redirect } from 'next/navigation';
import { prisma } from '@/lib/db/prisma';
import { audit } from '@/lib/auth/audit';
import { hashPassword, validatePassword, verifyPassword } from '@/lib/auth/password';
import { RATE_LIMITS, pruneRateLimits, rateLimit } from '@/lib/auth/rate-limit';
import {
  clearSessionCookie,
  readSessionCookie,
  setSessionCookie,
  signSession,
} from '@/lib/auth/session';
import { fieldErrors, signInSchema, signUpSchema } from '@/lib/validation';

export interface ActionState {
  ok: boolean;
  message?: string;
  errors?: Record<string, string>;
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
  pruneRateLimits();
  const key = await clientKey('signin');
  const limit = rateLimit(key, RATE_LIMITS.signIn.limit, RATE_LIMITS.signIn.windowMs);
  if (!limit.ok) {
    return {
      ok: false,
      message: `Too many sign-in attempts. Please wait about ${Math.ceil(limit.retryAfterSeconds / 60)} minutes and try again.`,
    };
  }

  const parsed = signInSchema.safeParse(Object.fromEntries(formData));
  if (!parsed.success) return { ok: false, errors: fieldErrors(parsed.error) };

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

  // Always run a comparison so that a missing account and a wrong password take
  // a similar amount of time.
  const hash =
    user?.passwordHash ?? '$2b$12$invalidinvalidinvalidinvalidinvalidinvalidinvalidinvalidin';
  const valid = await verifyPassword(parsed.data.password, hash);

  if (!user || !valid) {
    await audit({ userId: user?.id ?? null, action: 'auth.sign_in_failed' });
    return {
      ok: false,
      message: 'That email and password do not match an account. Please check both and try again.',
    };
  }

  await prisma.user.update({ where: { id: user.id }, data: { lastLoginAt: new Date() } });
  await audit({ userId: user.id, action: 'auth.sign_in' });
  await setSessionCookie(await signSession({ userId: user.id, tokenVersion: user.tokenVersion }));
  redirect(user.profile?.onboardingCompletedAt ? '/app' : '/app/onboarding');
}

export async function signOutAction(): Promise<void> {
  const session = await readSessionCookie();
  await audit({ userId: session?.userId ?? null, action: 'auth.sign_out' });
  await clearSessionCookie();
  redirect('/');
}
