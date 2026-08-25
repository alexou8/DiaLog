/**
 * Sign out every other device ("sign out everywhere else").
 *
 * The same shape as app/api/auth/password/route.ts, and for the same reason:
 * this bumps `tokenVersion`, so it revokes the cookie its own request arrived
 * with. That cannot be reported through the client router — see the header of
 * lib/auth/route-form.ts.
 *
 * This one was never measured failing, only because nothing had exercised it
 * ten times in a row. It was `useActionState` over a bumping Server Action,
 * which is exactly the shape that fails 6 times in 10 on the password form.
 */
import { type NextRequest, type NextResponse } from 'next/server';
import { prisma } from '@/lib/db/prisma';
import { audit } from '@/lib/auth/audit';
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

export type SessionsOutcome = 'revoked' | 'rate_limited';

export async function POST(request: NextRequest): Promise<NextResponse> {
  if (!isSameOrigin(request)) return crossOrigin();

  const session = await sessionFromRequest(request);
  if (!session) return toSignIn(request);

  const user = await prisma.user.findUnique({
    where: { id: session.userId },
    select: { id: true, tokenVersion: true },
  });
  if (!user || user.tokenVersion !== session.tokenVersion) return toSignIn(request);

  const limit = rateLimit(`revoke:${user.id}`, RATE_LIMITS.write.limit, RATE_LIMITS.write.windowMs);
  if (!limit.ok) return back(request, 'sessions', 'rate_limited');

  const updated = await prisma.user.update({
    where: { id: user.id },
    data: { tokenVersion: { increment: 1 } },
    select: { tokenVersion: true },
  });

  await audit({ userId: user.id, action: 'auth.sign_out_everywhere' });

  // Every cookie for this account is now stale, this browser's included. It
  // gets a replacement on the redirect; the other devices do not.
  return issueSession(back(request, 'sessions', 'revoked'), user.id, updated.tokenVersion);
}
