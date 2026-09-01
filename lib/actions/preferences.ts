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
import { verifyPassword } from '@/lib/auth/password';
import { RATE_LIMITS, rateLimit } from '@/lib/auth/rate-limit';
import { clearSessionCookie } from '@/lib/auth/session';
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
//
// Changing a password and "sign out everywhere" used to live here. Both bump
// `User.tokenVersion`, which revokes the cookie the request arrived with — an
// outcome a Server Action cannot deliver through the client router, and the
// cause of the change-password form failing 6 times in 10. They are now route
// handlers posted to by plain HTML forms:
//
//   app/api/auth/password/route.ts
//   app/api/auth/sessions/revoke/route.ts
//
// See lib/auth/route-form.ts for the full explanation, and
// tests/unit/auth/session-revocation.test.ts, which fails the build if a
// tokenVersion write reappears in this directory.

// --------------------------------------------------------------- deletion

/** Shared confirmation check for the two destructive actions below. */
// A 'use server' module may only export async functions, so this stays local.
const DELETE_PHRASE = 'DELETE';

async function checkDestructiveConfirmation(
  user: { id: string; email: string; passwordHash: string | null },
  formData: FormData,
): Promise<ActionState | null> {
  const confirmEmail = String(formData.get('confirmEmail') ?? '').trim();

  if (confirmEmail.toLowerCase() !== user.email.toLowerCase()) {
    return {
      ok: false,
      errors: { confirmEmail: 'Please type your account email address exactly to confirm.' },
    };
  }

  // Google-only accounts have no password to re-enter. Rather than send someone
  // off to set one before they can delete their own data, ask for a typed
  // phrase — the deliberate-action check the password was there to provide.
  if (user.passwordHash === null) {
    const phrase = String(formData.get('confirmPhrase') ?? '').trim();
    if (phrase !== DELETE_PHRASE) {
      return {
        ok: false,
        errors: { confirmPhrase: `Please type ${DELETE_PHRASE} exactly to confirm.` },
      };
    }
    return null;
  }

  const password = String(formData.get('password') ?? '');
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
    conversations,
    importBatches,
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
    // Health data that is derived from records rather than being one: an AI
    // conversation stores the question, the answer and the evidence findings
    // behind it, and an ImportBatch's issues keep the raw rejected rows
    // (ImportIssue.rawRow) of a failed import. Both cascade from their parent,
    // and both are health data the user reasonably expects "delete all my
    // records" to erase — leaving them behind meant a wiped account still held
    // free-text health discussion and raw imported rows.
    prisma.aIConversation.deleteMany({ where: { userId: user.id } }),
    prisma.importBatch.deleteMany({ where: { userId: user.id } }),
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
    note.count +
    conversations.count +
    importBatches.count;

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
