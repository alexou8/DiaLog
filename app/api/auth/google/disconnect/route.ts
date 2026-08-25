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
 */
import { NextResponse, type NextRequest } from 'next/server';
import { prisma } from '@/lib/db/prisma';
import { audit } from '@/lib/auth/audit';
import { readSessionCookie } from '@/lib/auth/session';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

function back(request: NextRequest, outcome: string): NextResponse {
  const url = request.nextUrl.clone();
  url.pathname = '/app/settings';
  url.search = `?unlinked=${outcome}`;
  // 303 so the browser follows with a GET, whatever it posted.
  return NextResponse.redirect(url, 303);
}

export async function POST(request: NextRequest): Promise<NextResponse> {
  // Server actions get CSRF protection from Next; a route handler has to do
  // its own. Only a form on this origin may disconnect an account.
  const origin = request.headers.get('origin');
  if (origin && origin !== request.nextUrl.origin) {
    return NextResponse.json({ error: 'cross_origin' }, { status: 403 });
  }

  const session = await readSessionCookie();
  if (!session) {
    const url = request.nextUrl.clone();
    url.pathname = '/sign-in';
    url.search = '';
    return NextResponse.redirect(url, 303);
  }

  const user = await prisma.user.findUnique({
    where: { id: session.userId },
    select: {
      id: true,
      tokenVersion: true,
      passwordHash: true,
      identities: { where: { provider: 'google' }, select: { id: true } },
    },
  });
  if (!user || user.tokenVersion !== session.tokenVersion) {
    const url = request.nextUrl.clone();
    url.pathname = '/sign-in';
    url.search = '';
    return NextResponse.redirect(url, 303);
  }

  const identity = user.identities[0];
  if (!identity) return back(request, 'absent');

  // Refused while Google is the only way in: removing it would lock the person
  // out of their own health records with no recovery path.
  if (user.passwordHash === null) return back(request, 'blocked');

  await prisma.authIdentity.delete({ where: { id: identity.id } });
  await audit({ userId: user.id, action: 'auth.google_unlinked' });

  return back(request, 'google');
}
