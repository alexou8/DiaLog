import { after } from 'next/server';
import { prisma } from '@/lib/db/prisma';
import { getEmailProvider } from '@/lib/email/provider';
import type { ActionState } from '@/lib/actions/auth';
import { fieldErrors, forgotPasswordSchema } from '@/lib/validation';
import { audit } from './audit';
import { isSessionSecretConfigured } from './session';
import { keyedRecoveryEmailHash } from './password-reset-email-key';
import { RATE_LIMITS, pruneRateLimits, rateLimit } from './rate-limit';
import {
  createResetToken,
  resetTokenHash,
  signResetProof,
  verifyResetProof,
} from './password-reset-token';

export const RESET_CONFIRMATION =
  'If an account with a password matches that email, we will send a reset link. If you use Google, continue with Google sign-in.';
export const RECOVERY_UNAVAILABLE =
  'Password recovery is temporarily unavailable. Please try again later.';

function recoveryOrigin(): string | null {
  try {
    // Resolve on the server at request time. Direct NEXT_PUBLIC_* accesses
    // are inlined by Next's build; CI and deployments reuse build artifacts
    // with different runtime origins, so a baked-in origin can misdirect mail.
    const { NEXT_PUBLIC_APP_URL } = process.env;
    const url = new URL(NEXT_PUBLIC_APP_URL ?? '');
    if (
      url.username ||
      url.password ||
      (url.protocol !== 'https:' &&
        !(url.protocol === 'http:' && ['localhost', '127.0.0.1', '[::1]'].includes(url.hostname)))
    )
      return null;
    return url.origin;
  } catch {
    return null;
  }
}

export async function requestPasswordReset(input: unknown, ip: string): Promise<ActionState> {
  if (!isSessionSecretConfigured()) {
    await audit({ action: 'auth.password_reset_failed' });
    return { ok: false, message: RECOVERY_UNAVAILABLE };
  }
  pruneRateLimits();
  const ipLimit = rateLimit(
    `pwreset-ip:${ip}`,
    RATE_LIMITS.passwordResetIp.limit,
    RATE_LIMITS.passwordResetIp.windowMs,
  );
  if (!ipLimit.ok) {
    await audit({ action: 'auth.password_reset_failed' });
    return { ok: false, message: 'Too many recovery requests. Please try again later.' };
  }
  const parsed = forgotPasswordSchema.safeParse(input);
  if (!parsed.success)
    return {
      ok: false,
      message: 'Please check your email address.',
      errors: fieldErrors(parsed.error),
    };

  const emailLimit = rateLimit(
    `pwreset-email:${keyedRecoveryEmailHash(parsed.data.email)}`,
    RATE_LIMITS.passwordResetEmail.limit,
    RATE_LIMITS.passwordResetEmail.windowMs,
  );
  // Recovery must not disclose account existence or authentication method even
  // though sign-up/sign-in do: this endpoint would otherwise become a targeted
  // mailbox-abuse oracle. Email throttling and delivery failures stay neutral too.
  const neutral: ActionState = { ok: true, message: RESET_CONFIRMATION };
  const origin = recoveryOrigin();
  let provider;
  try {
    provider = getEmailProvider();
    if (!origin || !provider.available()) throw new Error('Unavailable');
  } catch {
    await audit({ action: 'auth.password_reset_failed' });
    return { ok: false, message: RECOVERY_UNAVAILABLE };
  }
  if (!emailLimit.ok) {
    await audit({ action: 'auth.password_reset_failed' });
    return neutral;
  }
  try {
    const user = await prisma.user.findUnique({
      where: { email: parsed.data.email },
      select: { id: true, passwordHash: true },
    });
    // Next keeps after() callbacks alive through its request lifecycle. A bare
    // floating promise can be killed when a serverless response returns, silently
    // losing recovery mail. Defer audits, issuance and network delivery for ALL
    // account states so those account-dependent costs cannot time the response.
    after(async () => {
      let userId: string | undefined;
      try {
        userId = user?.id;
        await audit({ userId, action: 'auth.password_reset_requested' });
        if (!user || user.passwordHash === null) return;
        const token = createResetToken();
        const created = await prisma.$transaction(async (tx) => {
          // Issuance and consumption lock the same owner row, including when there
          // are no token rows yet. Two requests cannot leave two usable links behind.
          await tx.$queryRaw`SELECT id FROM "User" WHERE id = ${user.id} FOR UPDATE`;
          const current = await tx.user.findUnique({
            where: { id: user.id },
            select: { passwordHash: true },
          });
          if (!current?.passwordHash) return false;
          await tx.passwordResetToken.updateMany({
            where: { userId: user.id, usedAt: null },
            data: { usedAt: new Date() },
          });
          await tx.passwordResetToken.create({
            data: { userId: user.id, tokenHash: token.tokenHash, expiresAt: token.expiresAt },
          });
          return true;
        });
        if (!created) return;
        // Next serializes page URLs in RSC: a redirect-only entry keeps the raw
        // token out of rendered responses as well as out of our own form props.
        const url = new URL('/api/auth/password/reset', origin!);
        url.searchParams.set('token', token.raw);
        try {
          await provider.sendRecovery({ to: parsed.data.email, url: url.toString() });
        } catch {
          await prisma.passwordResetToken.updateMany({
            where: { userId: user.id, tokenHash: token.tokenHash, usedAt: null },
            data: { usedAt: new Date() },
          });
          throw new Error('Delivery failed');
        }
      } catch {
        await audit({ userId, action: 'auth.password_reset_failed' });
        console.error('Password recovery request failed. Check database and email delivery.');
      }
    });
  } catch {
    await audit({ action: 'auth.password_reset_failed' });
    // Fixed operational signal, never the thrown transport/Prisma error. Mail
    // failure is loud to operators without becoming an account-existence oracle.
    console.error('Password recovery request failed. Check database and email delivery.');
  }
  return neutral;
}

export async function prepareResetProof(raw: unknown): Promise<string | null> {
  if (!isSessionSecretConfigured()) return null;
  const tokenHash = resetTokenHash(raw);
  if (!tokenHash) return null;
  const token = await prisma.passwordResetToken.findUnique({
    where: { tokenHash },
    include: { user: { select: { passwordHash: true } } },
  });
  // The unguessable token is the recovery credential; no client-supplied user
  // id selects the account. All subsequent queries use its proven owner.
  if (!token || token.usedAt || token.expiresAt <= new Date() || token.user.passwordHash === null)
    return null;
  return signResetProof({ userId: token.userId, tokenHash }, token.expiresAt);
}

export async function isResetProofUsable(proof: string): Promise<boolean> {
  const claims = await verifyResetProof(proof);
  if (!claims) return false;
  return !!(await prisma.passwordResetToken.findFirst({
    where: {
      ...claims,
      usedAt: null,
      expiresAt: { gt: new Date() },
      user: { passwordHash: { not: null } },
    },
    select: { id: true },
  }));
}

// Reset tokens are account credentials, not health data. They deliberately
// survive deleteAllRecordsAction; PasswordResetToken.user still cascades when
// the account itself is deleted. No schema/migration change is required.
