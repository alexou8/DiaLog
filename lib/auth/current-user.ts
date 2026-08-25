import { cache } from 'react';
import { redirect } from 'next/navigation';
import type { Profile, User } from '@prisma/client';
import { prisma } from '@/lib/db/prisma';
import { readSessionCookie } from './session';

export type CurrentUser = User & { profile: Profile };

/**
 * Resolve the signed-in user for the current request. Deduped per request so
 * that layout, page and API handlers share a single query.
 */
export const getCurrentUser = cache(async (): Promise<CurrentUser | null> => {
  const session = await readSessionCookie();
  if (!session) return null;
  const user = await prisma.user.findUnique({
    where: { id: session.userId },
    include: { profile: true },
  });
  if (!user || !user.profile) return null;
  // Reject cookies minted before a credential change.
  if (user.tokenVersion !== session.tokenVersion) return null;
  return user as CurrentUser;
});

/** Server-component guard: redirects unauthenticated visitors to sign-in. */
export async function requireUser(): Promise<CurrentUser> {
  const user = await getCurrentUser();
  if (!user) redirect('/sign-in');
  return user;
}

/**
 * Guard for pages that need a completed profile. Kept out of the layout file
 * because Next.js route files may only export their known entry points.
 */
export async function requireOnboardedUser(): Promise<CurrentUser> {
  const user = await requireUser();
  if (!user.profile.onboardingCompletedAt) redirect('/app/onboarding');
  return user;
}
