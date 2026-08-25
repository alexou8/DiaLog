/**
 * Disconnect Google from the signed-in account.
 *
 * A route handler rather than a server action, and posted to by a plain HTML
 * form. Every outcome is a redirect back to Settings with a notice, and a
 * redirect issued from here is followed by the browser itself.
 *
 * That last part is the reason this is not a server action: an action that
 * redirects to the route it was submitted from — the settings page, differing
 * only in query string — is intermittently dropped by the client router. It
 * failed roughly three times in ten while the same code redirecting to any
 * other route passed forty times running, leaving the row deleted, the server
 * answering 303, and the person looking at a disabled "Disconnecting…" button.
 * Connecting already works this way (`../start`), so the two halves now match.
 *
 * The origin check, the 303 helpers and the session read are shared with the
 * password and sign-out-everywhere handlers — see lib/auth/route-form.ts,
 * which also explains why the whole settings surface works this way.
 */
import { type NextRequest, type NextResponse } from 'next/server';
import { prisma } from '@/lib/db/prisma';
import { audit } from '@/lib/auth/audit';
import {
  back,
  crossOrigin,
  isSameOrigin,
  sessionFromRequest,
  toSignIn,
} from '@/lib/auth/route-form';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export type UnlinkOutcome = 'google' | 'absent' | 'blocked';

function result(request: NextRequest, outcome: UnlinkOutcome): NextResponse {
  return back(request, 'unlinked', outcome);
}

export async function POST(request: NextRequest): Promise<NextResponse> {
  if (!isSameOrigin(request)) return crossOrigin();

  const session = await sessionFromRequest(request);
  if (!session) return toSignIn(request);

  const user = await prisma.user.findUnique({
    where: { id: session.userId },
    select: {
      id: true,
      tokenVersion: true,
      passwordHash: true,
      identities: { where: { provider: 'google' }, select: { id: true } },
    },
  });
  if (!user || user.tokenVersion !== session.tokenVersion) return toSignIn(request);

  const identity = user.identities[0];
  if (!identity) return result(request, 'absent');

  // Refused while Google is the only way in: removing it would lock the person
  // out of their own health records with no recovery path.
  if (user.passwordHash === null) return result(request, 'blocked');

  await prisma.authIdentity.delete({ where: { id: identity.id } });
  await audit({ userId: user.id, action: 'auth.google_unlinked' });

  return result(request, 'google');
}
