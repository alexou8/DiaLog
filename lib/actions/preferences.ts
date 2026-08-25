'use server';

/**
 * Settings-surface server actions: preferences, password, sessions and
 * account/data deletion.
 *
 * Every action here re-derives the user from the session cookie via
 * `requireUser()` — a userId is never accepted from form input — so nothing
 * here can be pointed at another account by a crafted request.
 */
import { revalidatePath } from 'next/cache';
import { redirect } from 'next/navigation';
import { prisma } from '@/lib/db/prisma';
import { requireUser } from '@/lib/auth/current-user';
import { audit } from '@/lib/auth/audit';
import { hashPassword, validatePassword, verifyPassword } from '@/lib/auth/password';
import { RATE_LIMITS, rateLimit } from '@/lib/auth/rate-limit';
import { clearSessionCookie, setSessionCookie, signSession } from '@/lib/auth/session';
import { GLUCOSE_ENTRY_BOUNDS, toMgdl } from '@/lib/domain/units';
import { fieldErrors, preferencesSchema } from '@/lib/validation';
import type { GlucoseUnit } from '@prisma/client';

export interface ActionState {
  ok: boolean;
  message?: string;
  errors?: Record<string, string>;
}

/** Very light guard against accidental hammering; these are low-frequency actions. */
function guard(userId: string, key: string): ActionState | null {
  const limit = rateLimit(`${key}:${userId}`, RATE_LIMITS.write.limit, RATE_LIMITS.write.windowMs);
  return limit.ok ? null : { ok: false, message: 'Please wait a moment and try again.' };
}

// ------------------------------------------------------------ preferences

export async function updatePreferencesAction(
  _prev: ActionState | null,
  formData: FormData,
): Promise<ActionState> {
  const user = await requireUser();
  const limited = guard(user.id, 'prefs');
  if (limited) return limited;

  const parsed = preferencesSchema.safeParse(Object.fromEntries(formData));
  if (!parsed.success) return { ok: false, errors: fieldErrors(parsed.error) };

  const d = parsed.data;
  const unit = d.glucoseUnit as GlucoseUnit;
  const bounds = GLUCOSE_ENTRY_BOUNDS[unit];

  if (d.targetLow < bounds.min || d.targetLow > bounds.max) {
    return {
      ok: false,
      errors: { targetLow: `Please enter a number between ${bounds.min} and ${bounds.max}.` },
    };
  }
  if (d.targetHigh < bounds.min || d.targetHigh > bounds.max) {
    return {
      ok: false,
      errors: { targetHigh: `Please enter a number between ${bounds.min} and ${bounds.max}.` },
    };
  }
  if (d.targetLow >= d.targetHigh) {
    return {
      ok: false,
      errors: {
        targetHigh: 'The top of your range needs to be higher than the bottom of your range.',
      },
    };
  }

  const targetLowMgdl = toMgdl(d.targetLow, unit);
  const targetHighMgdl = toMgdl(d.targetHigh, unit);

  const wantsExternalConsent = d.externalAiConsent;
  const hadConsent = user.profile.externalAiConsentAt != null;

  await prisma.profile.update({
    where: { userId: user.id },
    data: {
      displayName: d.displayName?.trim() || null,
      glucoseUnit: unit,
      locale: d.locale,
      timezone: d.timezone,
      targetLowMgdl,
      targetHighMgdl,
      detailLevel: d.detailLevel,
      largeText: d.largeText,
      reduceMotion: d.reduceMotion,
      aiEnabled: d.aiEnabled,
      // Consent is a timestamp, not a boolean: granting it records when, and
      // withdrawing it clears the record entirely rather than flipping a flag.
      externalAiConsentAt: wantsExternalConsent
        ? hadConsent
          ? user.profile.externalAiConsentAt
          : new Date()
        : null,
    },
  });

  await audit({ userId: user.id, action: 'preferences.update' });
  revalidatePath('/app', 'layout');

  return { ok: true, message: 'Your settings have been saved.' };
}

// ------------------------------------------------------------------ auth

export async function changePasswordAction(
  _prev: ActionState | null,
  formData: FormData,
): Promise<ActionState> {
  const user = await requireUser();
  const limited = guard(user.id, 'pwchange');
  if (limited) return limited;

  const currentPassword = String(formData.get('currentPassword') ?? '');
  const newPassword = String(formData.get('newPassword') ?? '');
  const confirmPassword = String(formData.get('confirmPassword') ?? '');

  if (!currentPassword) {
    return { ok: false, errors: { currentPassword: 'Please enter your current password.' } };
  }

  const valid = await verifyPassword(currentPassword, user.passwordHash);
  if (!valid) {
    await audit({ userId: user.id, action: 'auth.password_change_failed' });
    return {
      ok: false,
      errors: { currentPassword: 'That is not your current password. Please try again.' },
    };
  }

  const policy = validatePassword(newPassword);
  if (!policy.ok) return { ok: false, errors: { newPassword: policy.message } };

  if (newPassword !== confirmPassword) {
    return { ok: false, errors: { confirmPassword: 'The two new passwords do not match.' } };
  }

  const passwordHash = await hashPassword(newPassword);

  // Bump tokenVersion so every other signed-in device is logged out, then
  // immediately mint a fresh cookie for this device so the person making the
  // change is not logged out of the browser they are using right now.
  const updated = await prisma.user.update({
    where: { id: user.id },
    data: { passwordHash, tokenVersion: { increment: 1 } },
    select: { tokenVersion: true },
  });

  await audit({ userId: user.id, action: 'auth.password_change' });
  await setSessionCookie(
    await signSession({ userId: user.id, tokenVersion: updated.tokenVersion }),
  );

  return {
    ok: true,
    message: 'Your password has been changed. You have been signed out of any other devices.',
  };
}

export async function signOutEverywhereAction(
  _prev: ActionState | null,
  _formData: FormData,
): Promise<ActionState> {
  const user = await requireUser();

  const updated = await prisma.user.update({
    where: { id: user.id },
    data: { tokenVersion: { increment: 1 } },
    select: { tokenVersion: true },
  });

  await audit({ userId: user.id, action: 'auth.sign_out_everywhere' });
  await setSessionCookie(
    await signSession({ userId: user.id, tokenVersion: updated.tokenVersion }),
  );

  return {
    ok: true,
    message: 'Every other device has been signed out. This device stays signed in.',
  };
}

// --------------------------------------------------------------- deletion

/** Shared confirmation check for the two destructive actions below. */
async function checkDestructiveConfirmation(
  user: { id: string; email: string; passwordHash: string },
  formData: FormData,
): Promise<ActionState | null> {
  const confirmEmail = String(formData.get('confirmEmail') ?? '').trim();
  const password = String(formData.get('password') ?? '');

  if (confirmEmail.toLowerCase() !== user.email.toLowerCase()) {
    return {
      ok: false,
      errors: { confirmEmail: 'Please type your account email address exactly to confirm.' },
    };
  }
  if (!password) {
    return { ok: false, errors: { password: 'Please enter your password to confirm.' } };
  }
  const valid = await verifyPassword(password, user.passwordHash);
  if (!valid) {
    return { ok: false, errors: { password: 'That password is not correct.' } };
  }
  return null;
}

export async function deleteAllRecordsAction(
  _prev: ActionState | null,
  formData: FormData,
): Promise<ActionState> {
  const user = await requireUser();
  const limited = guard(user.id, 'deleteall');
  if (limited) return limited;

  const failure = await checkDestructiveConfirmation(user, formData);
  if (failure) return failure;

  await audit({ userId: user.id, action: 'data.delete_all_records' });

  const [
    glucose,
    meal,
    exercise,
    sleep,
    medication,
    weight,
    bloodPressure,
    hydration,
    symptom,
    mood,
    note,
  ] = await prisma.$transaction([
    prisma.glucoseReading.deleteMany({ where: { userId: user.id } }),
    prisma.meal.deleteMany({ where: { userId: user.id } }),
    prisma.exerciseSession.deleteMany({ where: { userId: user.id } }),
    prisma.sleepSession.deleteMany({ where: { userId: user.id } }),
    prisma.medicationEvent.deleteMany({ where: { userId: user.id } }),
    prisma.weightMeasurement.deleteMany({ where: { userId: user.id } }),
    prisma.bloodPressureMeasurement.deleteMany({ where: { userId: user.id } }),
    prisma.hydrationEvent.deleteMany({ where: { userId: user.id } }),
    prisma.symptomEntry.deleteMany({ where: { userId: user.id } }),
    prisma.moodEntry.deleteMany({ where: { userId: user.id } }),
    prisma.noteEntry.deleteMany({ where: { userId: user.id } }),
  ]);

  const total =
    glucose.count +
    meal.count +
    exercise.count +
    sleep.count +
    medication.count +
    weight.count +
    bloodPressure.count +
    hydration.count +
    symptom.count +
    mood.count +
    note.count;

  revalidatePath('/app', 'layout');

  return {
    ok: true,
    message: `${total} ${total === 1 ? 'record has' : 'records have'} been permanently deleted. Your account and settings were kept.`,
  };
}

export async function deleteAccountAction(
  _prev: ActionState | null,
  formData: FormData,
): Promise<ActionState> {
  const user = await requireUser();
  const limited = guard(user.id, 'deleteaccount');
  if (limited) return limited;

  const failure = await checkDestructiveConfirmation(user, formData);
  if (failure) return failure;

  // Audited before the row disappears — AuditEvent.userId is set null on
  // delete (see schema), so this is the last event that can name the account.
  await audit({ userId: user.id, action: 'auth.account_delete' });

  // Cascades: User -> Profile, Device, ImportBatch, every health record type,
  // Insight, AIConversation (-> AIMessage), PasswordResetToken all specify
  // onDelete: Cascade in prisma/schema.prisma, so this one delete removes
  // everything that belongs to the account.
  await prisma.user.delete({ where: { id: user.id } });

  await clearSessionCookie();
  redirect('/?deleted=1');
}
