'use server';

import { revalidatePath } from 'next/cache';
import { redirect } from 'next/navigation';
import type { GlucoseUnit } from '@prisma/client';
import { prisma } from '@/lib/db/prisma';
import { requireUser } from '@/lib/auth/current-user';
import { audit } from '@/lib/auth/audit';
import { RATE_LIMITS, rateLimit } from '@/lib/auth/rate-limit';
import { dedupeKey } from '@/lib/domain/dedupe';
import {
  isPlausibleGlucose,
  toMgdl,
  unitLabel,
  GLUCOSE_ENTRY_BOUNDS,
  KG_PER_LB,
  ML_PER_FL_OZ,
} from '@/lib/domain/units';
import { zonedDateToUtc } from '@/lib/domain/time';
import { deleteOwnedRecord, isRecordType } from '@/lib/db/health-records';
import {
  bloodPressureEntrySchema,
  exerciseEntrySchema,
  fieldErrors,
  glucoseEntrySchema,
  mealEntrySchema,
  medicationEntrySchema,
  moodEntrySchema,
  sleepEntrySchema,
  weightEntrySchema,
  hydrationEntrySchema,
  symptomEntrySchema,
} from '@/lib/validation';

export interface RecordActionState {
  ok: boolean;
  message?: string;
  errors?: Record<string, string>;
}

const DUPLICATE_MESSAGE =
  'You already have a record at that time with the same value, so nothing was added. Change the time if this is a separate entry.';

async function guardWrites(userId: string): Promise<RecordActionState | null> {
  const limit = rateLimit(`write:${userId}`, RATE_LIMITS.write.limit, RATE_LIMITS.write.windowMs);
  return limit.ok
    ? null
    : {
        ok: false,
        message: 'That is a lot of entries in a short time. Please wait a moment and try again.',
      };
}

/** Prisma's unique-constraint failure, used to turn duplicates into a friendly message. */
function isUniqueViolation(error: unknown): boolean {
  return (
    typeof error === 'object' &&
    error !== null &&
    'code' in error &&
    (error as { code: string }).code === 'P2002'
  );
}

export async function addGlucoseAction(
  _prev: RecordActionState | null,
  formData: FormData,
): Promise<RecordActionState> {
  const user = await requireUser();
  const limited = await guardWrites(user.id);
  if (limited) return limited;

  const parsed = glucoseEntrySchema.safeParse(Object.fromEntries(formData));
  if (!parsed.success) return { ok: false, errors: fieldErrors(parsed.error) };

  const { value, unit, takenAt, context, note } = parsed.data;
  if (!isPlausibleGlucose(value, unit as GlucoseUnit)) {
    const bounds = GLUCOSE_ENTRY_BOUNDS[unit as GlucoseUnit];
    return {
      ok: false,
      errors: {
        value: `Please enter a reading between ${bounds.min} and ${bounds.max} ${unitLabel(unit as GlucoseUnit)}. If that is really your reading, please double-check the unit.`,
      },
    };
  }

  const when = zonedDateToUtc(takenAt, user.profile.timezone);
  if (when.getTime() > Date.now() + 60_000) {
    return {
      ok: false,
      errors: { takenAt: 'That time is in the future. Please check the date and time.' },
    };
  }

  const valueMgdl = toMgdl(value, unit as GlucoseUnit);
  try {
    await prisma.glucoseReading.create({
      data: {
        userId: user.id,
        takenAt: when,
        valueMgdl,
        context,
        note: note?.trim() || null,
        source: 'MANUAL',
        dedupeKey: dedupeKey({ type: 'glucose', takenAt: when, value: valueMgdl }),
      },
    });
  } catch (error) {
    if (isUniqueViolation(error)) return { ok: false, message: DUPLICATE_MESSAGE };
    throw error;
  }

  await audit({ userId: user.id, action: 'record.create', entity: 'glucose' });
  revalidatePath('/app');
  revalidatePath('/app/glucose');
  redirect('/app/glucose?added=1');
}

export async function addMealAction(
  _prev: RecordActionState | null,
  formData: FormData,
): Promise<RecordActionState> {
  const user = await requireUser();
  const limited = await guardWrites(user.id);
  if (limited) return limited;

  const parsed = mealEntrySchema.safeParse(Object.fromEntries(formData));
  if (!parsed.success) return { ok: false, errors: fieldErrors(parsed.error) };

  const d = parsed.data;
  const when = zonedDateToUtc(d.takenAt, user.profile.timezone);
  try {
    await prisma.meal.create({
      data: {
        userId: user.id,
        takenAt: when,
        mealType: d.mealType,
        description: d.description,
        carbsG: d.carbsG,
        proteinG: d.proteinG,
        fatG: d.fatG,
        fiberG: d.fiberG,
        calories: d.calories,
        portion: d.portion?.trim() || null,
        note: d.note?.trim() || null,
        estimateSource: d.estimateSource,
        source: d.estimateSource === 'AI_ESTIMATE' ? 'AI_ASSISTED' : 'MANUAL',
        dedupeKey: dedupeKey({ type: 'meal', takenAt: when, discriminator: d.description }),
      },
    });
  } catch (error) {
    if (isUniqueViolation(error)) return { ok: false, message: DUPLICATE_MESSAGE };
    throw error;
  }

  await audit({ userId: user.id, action: 'record.create', entity: 'meal' });
  revalidatePath('/app');
  revalidatePath('/app/meals');
  redirect('/app/meals?added=1');
}

export async function addExerciseAction(
  _prev: RecordActionState | null,
  formData: FormData,
): Promise<RecordActionState> {
  const user = await requireUser();
  const limited = await guardWrites(user.id);
  if (limited) return limited;

  const parsed = exerciseEntrySchema.safeParse(Object.fromEntries(formData));
  if (!parsed.success) return { ok: false, errors: fieldErrors(parsed.error) };

  const d = parsed.data;
  const when = zonedDateToUtc(d.takenAt, user.profile.timezone);
  try {
    await prisma.exerciseSession.create({
      data: {
        userId: user.id,
        takenAt: when,
        endedAt: new Date(when.getTime() + d.durationMin * 60_000),
        activity: d.activity,
        durationMin: d.durationMin,
        intensity: d.intensity,
        distanceKm: d.distanceKm,
        steps: d.steps == null ? null : Math.round(d.steps),
        note: d.note?.trim() || null,
        dedupeKey: dedupeKey({
          type: 'exercise',
          takenAt: when,
          value: d.durationMin,
          discriminator: d.activity,
        }),
      },
    });
  } catch (error) {
    if (isUniqueViolation(error)) return { ok: false, message: DUPLICATE_MESSAGE };
    throw error;
  }

  await audit({ userId: user.id, action: 'record.create', entity: 'exercise' });
  revalidatePath('/app');
  revalidatePath('/app/activity');
  redirect('/app/activity?added=1');
}

export async function addSleepAction(
  _prev: RecordActionState | null,
  formData: FormData,
): Promise<RecordActionState> {
  const user = await requireUser();
  const limited = await guardWrites(user.id);
  if (limited) return limited;

  const parsed = sleepEntrySchema.safeParse(Object.fromEntries(formData));
  if (!parsed.success) return { ok: false, errors: fieldErrors(parsed.error) };

  const start = zonedDateToUtc(parsed.data.takenAt, user.profile.timezone);
  const end = zonedDateToUtc(parsed.data.endedAt, user.profile.timezone);
  if (end.getTime() <= start.getTime()) {
    return { ok: false, errors: { endedAt: 'The wake-up time needs to be after the bedtime.' } };
  }
  const durationMin = Math.round((end.getTime() - start.getTime()) / 60_000);
  if (durationMin > 24 * 60) {
    return {
      ok: false,
      errors: { endedAt: 'That is more than 24 hours. Please check the dates.' },
    };
  }

  try {
    await prisma.sleepSession.create({
      data: {
        userId: user.id,
        takenAt: start,
        endedAt: end,
        durationMin,
        quality: parsed.data.quality ?? null,
        note: parsed.data.note?.trim() || null,
        dedupeKey: dedupeKey({ type: 'sleep', takenAt: start, value: durationMin }),
      },
    });
  } catch (error) {
    if (isUniqueViolation(error)) return { ok: false, message: DUPLICATE_MESSAGE };
    throw error;
  }

  await audit({ userId: user.id, action: 'record.create', entity: 'sleep' });
  revalidatePath('/app');
  revalidatePath('/app/health');
  redirect('/app/health?added=sleep');
}

export async function addMedicationAction(
  _prev: RecordActionState | null,
  formData: FormData,
): Promise<RecordActionState> {
  const user = await requireUser();
  const limited = await guardWrites(user.id);
  if (limited) return limited;

  const parsed = medicationEntrySchema.safeParse(Object.fromEntries(formData));
  if (!parsed.success) return { ok: false, errors: fieldErrors(parsed.error) };

  const d = parsed.data;
  const when = zonedDateToUtc(d.takenAt, user.profile.timezone);
  try {
    await prisma.medicationEvent.create({
      data: {
        userId: user.id,
        takenAt: when,
        name: d.name,
        // Recorded exactly as the user typed it. DiaLog never interprets,
        // converts or calculates a dose.
        dose: d.dose?.trim() || null,
        route: d.route?.trim() || null,
        note: d.note?.trim() || null,
        dedupeKey: dedupeKey({ type: 'medication', takenAt: when, discriminator: d.name }),
      },
    });
  } catch (error) {
    if (isUniqueViolation(error)) return { ok: false, message: DUPLICATE_MESSAGE };
    throw error;
  }

  await audit({ userId: user.id, action: 'record.create', entity: 'medication' });
  revalidatePath('/app/health');
  redirect('/app/health?added=medication');
}

export async function addWeightAction(
  _prev: RecordActionState | null,
  formData: FormData,
): Promise<RecordActionState> {
  const user = await requireUser();
  const limited = await guardWrites(user.id);
  if (limited) return limited;

  const parsed = weightEntrySchema.safeParse(Object.fromEntries(formData));
  if (!parsed.success) return { ok: false, errors: fieldErrors(parsed.error) };

  const when = zonedDateToUtc(parsed.data.takenAt, user.profile.timezone);
  const weightKg = parsed.data.unit === 'LB' ? parsed.data.weight * KG_PER_LB : parsed.data.weight;
  if (weightKg < 20 || weightKg > 400) {
    return { ok: false, errors: { weight: 'Please check that number and the unit.' } };
  }

  try {
    await prisma.weightMeasurement.create({
      data: {
        userId: user.id,
        takenAt: when,
        weightKg,
        note: parsed.data.note?.trim() || null,
        dedupeKey: dedupeKey({ type: 'weight', takenAt: when, value: weightKg }),
      },
    });
  } catch (error) {
    if (isUniqueViolation(error)) return { ok: false, message: DUPLICATE_MESSAGE };
    throw error;
  }

  await audit({ userId: user.id, action: 'record.create', entity: 'weight' });
  revalidatePath('/app/health');
  redirect('/app/health?added=weight');
}

export async function addBloodPressureAction(
  _prev: RecordActionState | null,
  formData: FormData,
): Promise<RecordActionState> {
  const user = await requireUser();
  const limited = await guardWrites(user.id);
  if (limited) return limited;

  const parsed = bloodPressureEntrySchema.safeParse(Object.fromEntries(formData));
  if (!parsed.success) return { ok: false, errors: fieldErrors(parsed.error) };

  const d = parsed.data;
  if (d.diastolic >= d.systolic) {
    return {
      ok: false,
      errors: { diastolic: 'The lower number needs to be smaller than the upper number.' },
    };
  }
  const when = zonedDateToUtc(d.takenAt, user.profile.timezone);

  try {
    await prisma.bloodPressureMeasurement.create({
      data: {
        userId: user.id,
        takenAt: when,
        systolic: d.systolic,
        diastolic: d.diastolic,
        pulse: d.pulse,
        note: d.note?.trim() || null,
        dedupeKey: dedupeKey({
          type: 'bp',
          takenAt: when,
          value: d.systolic,
          discriminator: String(d.diastolic),
        }),
      },
    });
  } catch (error) {
    if (isUniqueViolation(error)) return { ok: false, message: DUPLICATE_MESSAGE };
    throw error;
  }

  await audit({ userId: user.id, action: 'record.create', entity: 'bloodPressure' });
  revalidatePath('/app/health');
  redirect('/app/health?added=bloodPressure');
}

export async function addMoodAction(
  _prev: RecordActionState | null,
  formData: FormData,
): Promise<RecordActionState> {
  const user = await requireUser();
  const limited = await guardWrites(user.id);
  if (limited) return limited;

  const parsed = moodEntrySchema.safeParse(Object.fromEntries(formData));
  if (!parsed.success) return { ok: false, errors: fieldErrors(parsed.error) };

  const when = zonedDateToUtc(parsed.data.takenAt, user.profile.timezone);
  try {
    await prisma.moodEntry.create({
      data: {
        userId: user.id,
        takenAt: when,
        mood: parsed.data.mood,
        stress: parsed.data.stress ?? null,
        note: parsed.data.note?.trim() || null,
        dedupeKey: dedupeKey({ type: 'mood', takenAt: when, value: parsed.data.mood }),
      },
    });
  } catch (error) {
    if (isUniqueViolation(error)) return { ok: false, message: DUPLICATE_MESSAGE };
    throw error;
  }

  await audit({ userId: user.id, action: 'record.create', entity: 'mood' });
  revalidatePath('/app/health');
  redirect('/app/health?added=mood');
}

/** Delete a single record. Ownership is enforced in the data layer. */
export async function deleteRecordAction(formData: FormData): Promise<void> {
  const user = await requireUser();
  const type = String(formData.get('type') ?? '');
  const id = String(formData.get('id') ?? '');
  if (!isRecordType(type) || !id) return;

  const deleted = await deleteOwnedRecord(user.id, type, id);
  if (deleted)
    await audit({ userId: user.id, action: 'record.delete', entity: type, entityId: id });

  revalidatePath('/app/history');
  revalidatePath('/app');
}

export async function addHydrationAction(
  _prev: RecordActionState | null,
  formData: FormData,
): Promise<RecordActionState> {
  const user = await requireUser();
  const limited = await guardWrites(user.id);
  if (limited) return limited;

  const parsed = hydrationEntrySchema.safeParse(Object.fromEntries(formData));
  if (!parsed.success) return { ok: false, errors: fieldErrors(parsed.error) };

  const when = zonedDateToUtc(parsed.data.takenAt, user.profile.timezone);
  // Canonical storage is millilitres; cups and fluid ounces are entry conveniences.
  const perUnit = { ML: 1, CUP: 250, FL_OZ: ML_PER_FL_OZ };
  const volumeMl = Math.round(parsed.data.volume * perUnit[parsed.data.unit]);
  if (volumeMl < 1 || volumeMl > 10_000) {
    return { ok: false, errors: { volume: 'Please check that amount and the unit.' } };
  }

  try {
    await prisma.hydrationEvent.create({
      data: {
        userId: user.id,
        takenAt: when,
        volumeMl,
        dedupeKey: dedupeKey({ type: 'hydration', takenAt: when, value: volumeMl }),
      },
    });
  } catch (error) {
    if (isUniqueViolation(error)) return { ok: false, message: DUPLICATE_MESSAGE };
    throw error;
  }

  await audit({ userId: user.id, action: 'record.create', entity: 'hydration' });
  revalidatePath('/app/health');
  redirect('/app/health?added=hydration');
}

export async function addSymptomAction(
  _prev: RecordActionState | null,
  formData: FormData,
): Promise<RecordActionState> {
  const user = await requireUser();
  const limited = await guardWrites(user.id);
  if (limited) return limited;

  const parsed = symptomEntrySchema.safeParse(Object.fromEntries(formData));
  if (!parsed.success) return { ok: false, errors: fieldErrors(parsed.error) };

  const when = zonedDateToUtc(parsed.data.takenAt, user.profile.timezone);
  try {
    await prisma.symptomEntry.create({
      data: {
        userId: user.id,
        takenAt: when,
        // Recorded exactly as described. DiaLog never interprets a symptom.
        symptom: parsed.data.symptom,
        severity: parsed.data.severity ?? null,
        note: parsed.data.note?.trim() || null,
        dedupeKey: dedupeKey({
          type: 'symptom',
          takenAt: when,
          discriminator: parsed.data.symptom,
        }),
      },
    });
  } catch (error) {
    if (isUniqueViolation(error)) return { ok: false, message: DUPLICATE_MESSAGE };
    throw error;
  }

  await audit({ userId: user.id, action: 'record.create', entity: 'symptom' });
  revalidatePath('/app/health');
  redirect('/app/health?added=symptom');
}
