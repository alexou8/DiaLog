/**
 * Set or change the signed-in account's password.
 *
 * A route handler posted to by a plain HTML form, not a Server Action, because
 * changing a password revokes the cookie the request arrived with — see the
 * header of lib/auth/route-form.ts for the failure that forces this shape.
 *
 * Every outcome, success or refusal, is a 303 back to Settings carrying a
 * `?password=<code>` the page renders as a sentence. No password, and nothing
 * derived from one, is ever put on the query string.
 */
import { type NextRequest, type NextResponse } from 'next/server';
import { prisma } from '@/lib/db/prisma';
import { audit } from '@/lib/auth/audit';
import { hashPassword, validatePassword, verifyPassword } from '@/lib/auth/password';
import { RATE_LIMITS, rateLimit } from '@/lib/auth/rate-limit';
import {
  back,
  crossOrigin,
  isSameOrigin,
  issueSession,
  sessionFromRequest,
  toSignIn,
} from '@/lib/auth/route-form';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

/** Outcome codes this handler can return. The page owns their wording. */
export type PasswordOutcome =
  | 'changed'
  | 'set'
  | 'missing_current'
  | 'wrong_current'
  | 'mismatch'
  | 'too_short'
  | 'too_long'
  | 'too_common'
  | 'rate_limited';

function result(request: NextRequest, outcome: PasswordOutcome): NextResponse {
  return back(request, 'password', outcome);
}

export async function POST(request: NextRequest): Promise<NextResponse> {
  if (!isSameOrigin(request)) return crossOrigin();

  const session = await sessionFromRequest(request);
  if (!session) return toSignIn(request);

  const user = await prisma.user.findUnique({
    where: { id: session.userId },
    select: { id: true, tokenVersion: true, passwordHash: true },
  });
  // Re-checked here rather than trusted from the cookie: a token whose version
  // has moved on is no longer a session, whatever it says about itself.
  if (!user || user.tokenVersion !== session.tokenVersion) return toSignIn(request);

  const limit = rateLimit(
    `pwchange:${user.id}`,
    RATE_LIMITS.write.limit,
    RATE_LIMITS.write.windowMs,
  );
  if (!limit.ok) return result(request, 'rate_limited');

  const form = await request.formData();
  const currentPassword = String(form.get('currentPassword') ?? '');
  const newPassword = String(form.get('newPassword') ?? '');
  const confirmPassword = String(form.get('confirmPassword') ?? '');

  // A Google-only account has no password to prove; the session cookie is the
  // proof. This is the "set a password" path, and it is what lets someone who
  // signed up with Google stop depending on Google.
  const existingHash = user.passwordHash;
  const settingFirstPassword = existingHash === null;

  if (existingHash !== null) {
    if (!currentPassword) return result(request, 'missing_current');
    if (!(await verifyPassword(currentPassword, existingHash))) {
      await audit({ userId: user.id, action: 'auth.password_change_failed' });
      return result(request, 'wrong_current');
    }
  }

  const policy = validatePassword(newPassword);
  if (!policy.ok) return result(request, policy.code);
  if (newPassword !== confirmPassword) return result(request, 'mismatch');

  const passwordHash = await hashPassword(newPassword);

  // Changing a password signs every other device out: bump tokenVersion, which
  // invalidates every outstanding cookie for the account at once.
  //
  // Setting a *first* password does not. Nothing was compromised, and there are
  // no password sessions to sign out — the account has only ever been reachable
  // through Google.
  const updated = await prisma.user.update({
    where: { id: user.id },
    data: settingFirstPassword
      ? { passwordHash }
      : { passwordHash, tokenVersion: { increment: 1 } },
    select: { tokenVersion: true },
  });

  await audit({
    userId: user.id,
    action: settingFirstPassword ? 'auth.password_set' : 'auth.password_change',
  });

  const response = result(request, settingFirstPassword ? 'set' : 'changed');
  // The bump above just invalidated this browser's cookie too. Replacing it on
  // the redirect itself is what keeps the person signed in on the device they
  // are using — the GET the browser makes next already carries the new token.
  return settingFirstPassword ? response : issueSession(response, user.id, updated.tokenVersion);
}
