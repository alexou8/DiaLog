'use server';

import { redirect } from 'next/navigation';
import { revalidatePath } from 'next/cache';
import { prisma } from '@/lib/db/prisma';
import { requireUser } from '@/lib/auth/current-user';
import { audit } from '@/lib/auth/audit';
import { isValidTimeZone } from '@/lib/domain/time';
import { fieldErrors, onboardingSchema } from '@/lib/validation';
import type { ActionState } from './auth';

type OnboardingState = ActionState & { values?: Record<string, string | string[]> };

/**
 * The submitted answers, echoed back on failure so a no-JavaScript submit
 * does not come back as an empty form. Same contract as the record actions.
 */
function submittedValues(formData: FormData): Record<string, string | string[]> {
  const out: Record<string, string | string[]> = {};
  for (const [key, value] of formData) {
    if (typeof value !== 'string' || key.startsWith('$')) continue;
    const existing = out[key];
    out[key] =
      existing === undefined
        ? value
        : Array.isArray(existing)
          ? [...existing, value]
          : [existing, value];
  }
  return out;
}

export async function completeOnboardingAction(
  _prev: OnboardingState | null,
  formData: FormData,
): Promise<OnboardingState> {
  const user = await requireUser();

  const parsed = onboardingSchema.safeParse({
    displayName: formData.get('displayName') ?? '',
    condition: formData.get('condition'),
    glucoseUnit: formData.get('glucoseUnit'),
    timezone: formData.get('timezone'),
    goals: formData.getAll('goals').map(String),
  });
  if (!parsed.success) {
    return { ok: false, errors: fieldErrors(parsed.error), values: submittedValues(formData) };
  }
  // The form now offers the browser's own zone even when it is not in the
  // curated list, so the only check left is the one that matters: every
  // reading's day and time of day is computed in this zone, and an unknown
  // one would throw on the first Intl call that used it.
  if (!isValidTimeZone(parsed.data.timezone)) {
    return {
      ok: false,
      errors: { timezone: 'Please choose your time zone from the list.' },
      values: submittedValues(formData),
    };
  }

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
