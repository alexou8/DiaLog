import { NextResponse, type NextRequest } from 'next/server';
import { prisma } from '@/lib/db/prisma';
import { audit } from '@/lib/auth/audit';
import { hashPassword, validatePassword, type PasswordPolicyCode } from '@/lib/auth/password';
import { verifyResetProof } from '@/lib/auth/password-reset-token';
import { crossOrigin, isSameOrigin } from '@/lib/auth/route-form';
import {
  isSessionSecretConfigured,
  SESSION_COOKIE,
  SESSION_COOKIE_OPTIONS,
} from '@/lib/auth/session';
import { pruneRateLimits, RATE_LIMITS, rateLimit } from '@/lib/auth/rate-limit';
import { resetPasswordSchema } from '@/lib/validation';
import { prepareResetProof } from '@/lib/auth/password-reset';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
export type ResetOutcome =
  | PasswordPolicyCode
  | 'mismatch'
  | 'invalid'
  | 'rate_limited'
  | 'unavailable';

/** A dedicated redirect helper: settings' back() must retain its destination. */
function result(
  request: NextRequest,
  outcome: ResetOutcome | 'reset',
  proof?: string,
): NextResponse {
  const url = new URL(outcome === 'reset' ? '/sign-in' : '/reset-password', request.url);
  url.searchParams.set('password', outcome);
  if (proof) url.searchParams.set('proof', proof);
  const response = NextResponse.redirect(url, 303);
  response.headers.set('Cache-Control', 'no-store');
  response.headers.set('Referrer-Policy', 'no-referrer');
  return response;
}

/** GET never consumes a token, so opening a link in a mail scanner is harmless. */
export async function GET(request: NextRequest): Promise<NextResponse> {
  try {
    if (!isSessionSecretConfigured()) return result(request, 'unavailable');
    const proof = await prepareResetProof(request.nextUrl.searchParams.get('token'));
    if (!proof) return result(request, 'invalid');
    const url = new URL('/reset-password', request.url);
    url.searchParams.set('proof', proof);
    const response = NextResponse.redirect(url, 303);
    response.headers.set('Cache-Control', 'no-store');
    response.headers.set('Referrer-Policy', 'no-referrer');
    return response;
  } catch {
    return result(request, 'unavailable');
  }
}

export async function POST(request: NextRequest): Promise<NextResponse> {
  // Recovery is bearer-authenticated, so SameSite session cookies offer no
  // CSRF fallback here. Require Origin as well as the shared same-origin check.
  if (!request.headers.get('origin') || !isSameOrigin(request)) {
    await audit({ action: 'auth.password_reset_failed' });
    return crossOrigin();
  }
  const failure = async (outcome: ResetOutcome, proof?: string, userId?: string) => {
    await audit({ userId, action: 'auth.password_reset_failed' });
    return result(request, outcome, proof);
  };
  if (!isSessionSecretConfigured()) return failure('unavailable');
  pruneRateLimits();
  const ip =
    request.headers.get('x-forwarded-for')?.split(',')[0]?.trim() ??
    request.headers.get('x-real-ip') ??
    'unknown';
  if (
    !rateLimit(
      `pwreset-submit:${ip}`,
      RATE_LIMITS.passwordResetSubmit.limit,
      RATE_LIMITS.passwordResetSubmit.windowMs,
    ).ok
  )
    return failure('rate_limited');
  let input;
  try {
    input = resetPasswordSchema.safeParse(Object.fromEntries(await request.formData()));
  } catch {
    return failure('invalid');
  }
  if (!input.success) return failure('invalid');
  const { proof, newPassword, confirmPassword } = input.data;
  const claims = await verifyResetProof(proof);
  if (!claims) return failure('invalid');
  const policy = validatePassword(newPassword);
  if (!policy.ok) return failure(policy.code, proof, claims.userId);
  if (newPassword !== confirmPassword) return failure('mismatch', proof, claims.userId);

  try {
    const passwordHash = await hashPassword(newPassword);
    const completed = await prisma.$transaction(async (tx) => {
      await tx.$queryRaw`SELECT id FROM "User" WHERE id = ${claims.userId} FOR UPDATE`;
      const user = await tx.user.findUnique({
        where: { id: claims.userId },
        select: { passwordHash: true },
      });
      if (!user?.passwordHash) return false;
      // Conditional consumption plus the owner lock makes a double submit
      // single-use, and serializes it with issuance of a replacement link.
      const consumed = await tx.passwordResetToken.updateMany({
        where: {
          userId: claims.userId,
          tokenHash: claims.tokenHash,
          usedAt: null,
          expiresAt: { gt: new Date() },
        },
        data: { usedAt: new Date() },
      });
      if (consumed.count !== 1) return false;
      await tx.user.update({
        where: { id: claims.userId },
        data: { passwordHash, tokenVersion: { increment: 1 } },
      });
      await tx.passwordResetToken.updateMany({
        where: { userId: claims.userId, usedAt: null },
        data: { usedAt: new Date() },
      });
      return true;
    });
    if (!completed) return failure('invalid', undefined, claims.userId);
    await audit({ userId: claims.userId, action: 'auth.password_reset_completed' });
    const response = result(request, 'reset');
    // No auto-sign-in: every old session is revoked, including this browser's.
    response.cookies.set(SESSION_COOKIE, '', { ...SESSION_COOKIE_OPTIONS, maxAge: 0 });
    return response;
  } catch {
    return failure('unavailable', proof, claims.userId);
  }
}
