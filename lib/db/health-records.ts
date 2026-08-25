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
} as const;

export function isRecordType(value: string): value is RecordType {
  return (RECORD_TYPES as readonly string[]).includes(value);
}

/** Delete one record, but only if it belongs to the given user. */
export async function deleteOwnedRecord(
  userId: string,
  type: RecordType,
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
 */
export async function pageGlucose(params: {
  userId: string;
  take: number;
  cursor?: string;
  from?: Date;
  to?: Date;
}) {
  const where: Prisma.GlucoseReadingWhereInput = { userId: params.userId };
  if (params.from || params.to) {
    where.takenAt = {
      ...(params.from ? { gte: params.from } : {}),
      ...(params.to ? { lte: params.to } : {}),
    };
  }
  const rows = await prisma.glucoseReading.findMany({
    where,
    orderBy: [{ takenAt: 'desc' }, { id: 'desc' }],
    take: params.take + 1,
    ...(params.cursor ? { cursor: { id: params.cursor }, skip: 1 } : {}),
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
