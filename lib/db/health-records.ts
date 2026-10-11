/**
 * User-scoped data access for health records.
 *
 * Every function here takes a `userId` and includes it in the query. Nothing in
 * the application reads a health record by id alone — the id is always paired
 * with the owner, so a guessed or leaked id from another account simply does
 * not match. This is the single place that rule needs to hold.
 */
import type { Prisma } from '@prisma/client';
import { prisma } from './prisma';

export const RECORD_TYPES = [
  'glucose',
  'meal',
  'exercise',
  'sleep',
  'medication',
  'weight',
  'bloodPressure',
  'mood',
] as const;

export type RecordType = (typeof RECORD_TYPES)[number];

/**
 * Every type a person can delete one record of. A superset of RECORD_TYPES:
 * hydration and symptom entries are logged from the Health page but have no
 * History tab, and RECORD_TYPES is also the History tab list (the page types
 * its labels as `Record<RecordType, string>`), so they are added here rather
 * than there. Before this, a hydration or symptom entry could be created but
 * never individually removed.
 */
export const DELETABLE_RECORD_TYPES = [...RECORD_TYPES, 'hydration', 'symptom'] as const;

export type DeletableRecordType = (typeof DELETABLE_RECORD_TYPES)[number];

/** Prisma delegate for each record type, keyed by the type name used in URLs. */
const DELEGATES = {
  glucose: () => prisma.glucoseReading,
  meal: () => prisma.meal,
  exercise: () => prisma.exerciseSession,
  sleep: () => prisma.sleepSession,
  medication: () => prisma.medicationEvent,
  weight: () => prisma.weightMeasurement,
  bloodPressure: () => prisma.bloodPressureMeasurement,
  mood: () => prisma.moodEntry,
  hydration: () => prisma.hydrationEvent,
  symptom: () => prisma.symptomEntry,
} as const satisfies Record<DeletableRecordType, () => unknown>;

export function isRecordType(value: string): value is RecordType {
  return (RECORD_TYPES as readonly string[]).includes(value);
}

/** As isRecordType, for the wider set deleteOwnedRecord accepts. */
export function isDeletableRecordType(value: string): value is DeletableRecordType {
  return (DELETABLE_RECORD_TYPES as readonly string[]).includes(value);
}

/** Delete one record, but only if it belongs to the given user. */
export async function deleteOwnedRecord(
  userId: string,
  type: DeletableRecordType,
  id: string,
): Promise<boolean> {
  const delegate = DELEGATES[type]();
  // deleteMany with both ids means an id belonging to someone else deletes nothing.
  const result = await (
    delegate as { deleteMany: (args: unknown) => Promise<{ count: number }> }
  ).deleteMany({
    where: { id, userId },
  });
  return result.count > 0;
}

export interface WindowArgs {
  userId: string;
  from: Date;
  to: Date;
}

export function glucoseInWindow({ userId, from, to }: WindowArgs) {
  return prisma.glucoseReading.findMany({
    where: { userId, takenAt: { gte: from, lte: to } },
    orderBy: { takenAt: 'asc' },
    select: { id: true, takenAt: true, valueMgdl: true, context: true, note: true, source: true },
  });
}

export function mealsInWindow({ userId, from, to }: WindowArgs) {
  return prisma.meal.findMany({
    where: { userId, takenAt: { gte: from, lte: to } },
    orderBy: { takenAt: 'asc' },
    select: { id: true, takenAt: true, mealType: true, carbsG: true, description: true },
  });
}

export function exerciseInWindow({ userId, from, to }: WindowArgs) {
  return prisma.exerciseSession.findMany({
    where: { userId, takenAt: { gte: from, lte: to } },
    orderBy: { takenAt: 'asc' },
    select: {
      id: true,
      takenAt: true,
      endedAt: true,
      durationMin: true,
      activity: true,
      intensity: true,
    },
  });
}

export function sleepInWindow({ userId, from, to }: WindowArgs) {
  return prisma.sleepSession.findMany({
    where: { userId, takenAt: { gte: from, lte: to } },
    orderBy: { takenAt: 'asc' },
    select: { id: true, takenAt: true, endedAt: true, durationMin: true, quality: true },
  });
}

export function medicationsInWindow({ userId, from, to }: WindowArgs) {
  return prisma.medicationEvent.findMany({
    where: { userId, takenAt: { gte: from, lte: to } },
    orderBy: { takenAt: 'asc' },
    select: { id: true, takenAt: true, name: true },
  });
}

export function moodsInWindow({ userId, from, to }: WindowArgs) {
  return prisma.moodEntry.findMany({
    where: { userId, takenAt: { gte: from, lte: to } },
    orderBy: { takenAt: 'asc' },
    select: { id: true, takenAt: true, mood: true, stress: true },
  });
}

/**
 * Everything the analytics engine needs for a window, in one round trip.
 * Deliberately selects only the columns the analysis uses — notes and raw
 * payloads never leave the database for an analysis run.
 */
export async function loadAnalyticsWindow(args: WindowArgs) {
  const [glucose, meals, exercise, sleep, medications, moods] = await Promise.all([
    glucoseInWindow(args),
    mealsInWindow(args),
    exerciseInWindow(args),
    sleepInWindow(args),
    medicationsInWindow(args),
    moodsInWindow(args),
  ]);
  return { glucose, meals, exercise, sleep, medications, moods };
}

/** Counts used for empty states, onboarding progress and data-quality reporting. */
export async function recordCounts(userId: string): Promise<Record<RecordType, number>> {
  const [glucose, meal, exercise, sleep, medication, weight, bloodPressure, mood] =
    await Promise.all([
      prisma.glucoseReading.count({ where: { userId } }),
      prisma.meal.count({ where: { userId } }),
      prisma.exerciseSession.count({ where: { userId } }),
      prisma.sleepSession.count({ where: { userId } }),
      prisma.medicationEvent.count({ where: { userId } }),
      prisma.weightMeasurement.count({ where: { userId } }),
      prisma.bloodPressureMeasurement.count({ where: { userId } }),
      prisma.moodEntry.count({ where: { userId } }),
    ]);
  return { glucose, meal, exercise, sleep, medication, weight, bloodPressure, mood };
}

/**
 * Paginated history across a single record type. Keyset pagination on
 * (takenAt, id) so that deep pages stay fast on large histories.
 *
 * The cursor arrives from `?cursor=` in the URL, so it is client-supplied and
 * is resolved scoped by owner like any other id. This used Prisma's
 * `cursor: { id }`, which looks the cursor row up by id alone — outside the
 * `userId` filter — so another account's reading id positioned this user's
 * page at that reading's timestamp: an ordering oracle on someone else's
 * data. An unknown or foreign cursor now simply restarts from the first page,
 * which is also the kind outcome for a cursor whose reading was deleted.
 */
export async function pageGlucose(params: {
  userId: string;
  take: number;
  cursor?: string;
  from?: Date;
  to?: Date;
}) {
  const conditions: Prisma.GlucoseReadingWhereInput[] = [{ userId: params.userId }];
  if (params.from || params.to) {
    conditions.push({
      takenAt: {
        ...(params.from ? { gte: params.from } : {}),
        ...(params.to ? { lte: params.to } : {}),
      },
    });
  }
  const after = params.cursor
    ? await prisma.glucoseReading.findFirst({
        where: { id: params.cursor, userId: params.userId },
        select: { id: true, takenAt: true },
      })
    : null;
  if (after) {
    // Strictly after the cursor in (takenAt desc, id desc) order.
    conditions.push({
      OR: [{ takenAt: { lt: after.takenAt } }, { takenAt: after.takenAt, id: { lt: after.id } }],
    });
  }
  const rows = await prisma.glucoseReading.findMany({
    where: { AND: conditions },
    orderBy: [{ takenAt: 'desc' }, { id: 'desc' }],
    take: params.take + 1,
    include: {
      importBatch: { select: { id: true, filename: true, connectorName: true } },
      device: { select: { label: true } },
    },
  });
  const hasMore = rows.length > params.take;
  return {
    rows: hasMore ? rows.slice(0, params.take) : rows,
    nextCursor: hasMore ? rows[params.take - 1]?.id : undefined,
  };
}

/**
 * Erase every health record and everything derived from one, for one user,
 * keeping the account, profile and credentials. The engine behind "Delete all
 * my records" (lib/actions/preferences.ts), kept here rather than in that file
 * because a 'use server' module turns every exported async function into a
 * client-callable endpoint — exporting a `(userId) => …` from there would let
 * any signed-in client erase any account. Here it is only reachable through
 * callers that derive `userId` from the session.
 *
 * Per the Deletion invariant (AGENTS.md), each model below is listed with the
 * reason it is in scope; a new model derived from health data belongs here.
 * Returns the number of rows removed per model, in one transaction so a
 * failure leaves nothing half-deleted.
 */
export async function deleteAllHealthData(userId: string) {
  const where = { userId };
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
    insights,
    devices,
  ] = await prisma.$transaction([
    // The eleven record tables. FoodItem cascades from Meal.
    prisma.glucoseReading.deleteMany({ where }),
    prisma.meal.deleteMany({ where }),
    prisma.exerciseSession.deleteMany({ where }),
    prisma.sleepSession.deleteMany({ where }),
    prisma.medicationEvent.deleteMany({ where }),
    prisma.weightMeasurement.deleteMany({ where }),
    prisma.bloodPressureMeasurement.deleteMany({ where }),
    prisma.hydrationEvent.deleteMany({ where }),
    prisma.symptomEntry.deleteMany({ where }),
    prisma.moodEntry.deleteMany({ where }),
    prisma.noteEntry.deleteMany({ where }),
    // Health data that is derived from records rather than being one: an AI
    // conversation stores the question, the answer and the evidence findings
    // behind it, and an ImportBatch's issues keep the raw rejected rows
    // (ImportIssue.rawRow) of a failed import. Both cascade to their children.
    prisma.aIConversation.deleteMany({ where }),
    prisma.importBatch.deleteMany({ where }),
    // Insight.summary/detail/evidence are statements about the user's
    // readings ("your post-dinner readings run higher..."). Nothing writes the
    // table today, which is exactly when a gap like this gets missed.
    prisma.insight.deleteMany({ where }),
    // A Device row names the meter or CGM the person uses (label, vendor,
    // model), which itself says something about their condition and treatment.
    // Readings and batches point at it with ON DELETE SET NULL, and both are
    // already gone above, so nothing is orphaned.
    prisma.device.deleteMany({ where }),
  ]);
  return {
    glucose: glucose.count,
    meal: meal.count,
    exercise: exercise.count,
    sleep: sleep.count,
    medication: medication.count,
    weight: weight.count,
    bloodPressure: bloodPressure.count,
    hydration: hydration.count,
    symptom: symptom.count,
    mood: mood.count,
    note: note.count,
    conversations: conversations.count,
    importBatches: importBatches.count,
    insights: insights.count,
    devices: devices.count,
  };
}
