'use server';

import { redirect } from 'next/navigation';
import { revalidatePath } from 'next/cache';
import { prisma } from '@/lib/db/prisma';
import { requireUser } from '@/lib/auth/current-user';
import { audit } from '@/lib/auth/audit';
import { fieldErrors, onboardingSchema } from '@/lib/validation';
import type { ActionState } from './auth';

export async function completeOnboardingAction(
  _prev: ActionState | null,
  formData: FormData,
): Promise<ActionState> {
  const user = await requireUser();

  const parsed = onboardingSchema.safeParse({
    displayName: formData.get('displayName') ?? '',
    condition: formData.get('condition'),
    glucoseUnit: formData.get('glucoseUnit'),
    timezone: formData.get('timezone'),
    goals: formData.getAll('goals').map(String),
  });
  if (!parsed.success) return { ok: false, errors: fieldErrors(parsed.error) };

  await prisma.profile.update({
    where: { userId: user.id },
    data: {
      displayName: parsed.data.displayName?.trim() || null,
      condition: parsed.data.condition,
      glucoseUnit: parsed.data.glucoseUnit,
      timezone: parsed.data.timezone,
      goals: parsed.data.goals,
      onboardingCompletedAt: new Date(),
    },
  });

  await audit({ userId: user.id, action: 'profile.onboarding_complete' });
  revalidatePath('/app', 'layout');
  redirect('/app');
}

/** Lets someone finish setting up later without being trapped in onboarding. */
export async function skipOnboardingAction(): Promise<void> {
  const user = await requireUser();
  await prisma.profile.update({
    where: { userId: user.id },
    data: { onboardingCompletedAt: new Date() },
  });
  revalidatePath('/app', 'layout');
  redirect('/app');
}
