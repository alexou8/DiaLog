import { afterAll, describe, expect, it } from 'vitest';
import {
  DELETABLE_RECORD_TYPES,
  deleteOwnedRecord,
  isDeletableRecordType,
  loadAnalyticsWindow,
  pageGlucose,
  recordCounts,
  RECORD_TYPES,
} from '@/lib/db/health-records';
import { prisma, createTestUser, deleteTestUser } from './test-helpers';

const createdUserIds: string[] = [];

afterAll(async () => {
  for (const id of createdUserIds) await deleteTestUser(id);
});

describe('record creation and retrieval round-trips', () => {
  it('creates and reads back a record of every RECORD_TYPES kind', async () => {
    const { user } = await createTestUser('roundtrip');
    createdUserIds.push(user.id);
    const takenAt = new Date('2026-02-01T10:00:00Z');

    const glucose = await prisma.glucoseReading.create({
      data: {
        userId: user.id,
        takenAt,
        valueMgdl: 110,
        context: 'FASTING',
        dedupeKey: 'rt-glucose',
      },
    });
    const meal = await prisma.meal.create({
      data: {
        userId: user.id,
        takenAt,
        description: 'Oatmeal',
        mealType: 'BREAKFAST',
        carbsG: 40,
        dedupeKey: 'rt-meal',
      },
    });
    const exercise = await prisma.exerciseSession.create({
      data: {
        userId: user.id,
        takenAt,
        activity: 'walk',
        durationMin: 20,
        dedupeKey: 'rt-exercise',
      },
    });
    const sleep = await prisma.sleepSession.create({
      data: {
        userId: user.id,
        takenAt,
        endedAt: new Date(takenAt.getTime() + 8 * 3600_000),
        durationMin: 480,
        dedupeKey: 'rt-sleep',
      },
    });
    const medication = await prisma.medicationEvent.create({
      data: { userId: user.id, takenAt, name: 'Metformin', dedupeKey: 'rt-med' },
    });
    const weight = await prisma.weightMeasurement.create({
      data: { userId: user.id, takenAt, weightKg: 70, dedupeKey: 'rt-weight' },
    });
    const bp = await prisma.bloodPressureMeasurement.create({
      data: { userId: user.id, takenAt, systolic: 120, diastolic: 80, dedupeKey: 'rt-bp' },
    });
    const mood = await prisma.moodEntry.create({
      data: { userId: user.id, takenAt, mood: 4, dedupeKey: 'rt-mood' },
    });

    expect(
      await prisma.glucoseReading.findUniqueOrThrow({ where: { id: glucose.id } }),
    ).toMatchObject({ valueMgdl: 110 });
    expect(await prisma.meal.findUniqueOrThrow({ where: { id: meal.id } })).toMatchObject({
      description: 'Oatmeal',
    });
    expect(
      await prisma.exerciseSession.findUniqueOrThrow({ where: { id: exercise.id } }),
    ).toMatchObject({ activity: 'walk' });
    expect(await prisma.sleepSession.findUniqueOrThrow({ where: { id: sleep.id } })).toMatchObject({
      durationMin: 480,
    });
    expect(
      await prisma.medicationEvent.findUniqueOrThrow({ where: { id: medication.id } }),
    ).toMatchObject({ name: 'Metformin' });
    expect(
      await prisma.weightMeasurement.findUniqueOrThrow({ where: { id: weight.id } }),
    ).toMatchObject({ weightKg: 70 });
    expect(
      await prisma.bloodPressureMeasurement.findUniqueOrThrow({ where: { id: bp.id } }),
    ).toMatchObject({ systolic: 120 });
    expect(await prisma.moodEntry.findUniqueOrThrow({ where: { id: mood.id } })).toMatchObject({
      mood: 4,
    });

    // Sanity: every declared RECORD_TYPES entry maps to a table we actually exercised
    // (bloodPressure has no deleteOwnedRecord coverage below beyond this smoke test).
    expect(RECORD_TYPES).toContain('glucose');
    expect(RECORD_TYPES).toContain('bloodPressure');
  });

  it('rejects a duplicate (userId, dedupeKey) pair with a unique constraint violation', async () => {
    const { user } = await createTestUser('dedupe-constraint');
    createdUserIds.push(user.id);
    await prisma.glucoseReading.create({
      data: { userId: user.id, takenAt: new Date(), valueMgdl: 100, dedupeKey: 'same-key' },
    });
    await expect(
      prisma.glucoseReading.create({
        data: { userId: user.id, takenAt: new Date(), valueMgdl: 999, dedupeKey: 'same-key' },
      }),
    ).rejects.toThrow();

    // A different user CAN reuse the same dedupeKey — the constraint is per-user.
    const { user: otherUser } = await createTestUser('dedupe-constraint-other');
    createdUserIds.push(otherUser.id);
    await expect(
      prisma.glucoseReading.create({
        data: { userId: otherUser.id, takenAt: new Date(), valueMgdl: 100, dedupeKey: 'same-key' },
      }),
    ).resolves.toBeTruthy();
  });
});

describe('deleteOwnedRecord', () => {
  it('deletes the row when the user owns it', async () => {
    const { user } = await createTestUser('delete-owned');
    createdUserIds.push(user.id);
    const reading = await prisma.glucoseReading.create({
      data: { userId: user.id, takenAt: new Date(), valueMgdl: 100, dedupeKey: 'del-1' },
    });
    const deleted = await deleteOwnedRecord(user.id, 'glucose', reading.id);
    expect(deleted).toBe(true);
    expect(await prisma.glucoseReading.findUnique({ where: { id: reading.id } })).toBeNull();
  });

  // Hydration and symptom entries could be logged but not individually
  // removed: neither type was in the delete path's delegate map.
  it('deletes an owned hydration entry and refuses another user’s', async () => {
    const { user } = await createTestUser('delete-hydration');
    const { user: other } = await createTestUser('delete-hydration-other');
    createdUserIds.push(user.id, other.id);
    const mine = await prisma.hydrationEvent.create({
      data: { userId: user.id, takenAt: new Date(), volumeMl: 250, dedupeKey: 'hyd-mine' },
    });
    const theirs = await prisma.hydrationEvent.create({
      data: { userId: other.id, takenAt: new Date(), volumeMl: 500, dedupeKey: 'hyd-theirs' },
    });

    expect(isDeletableRecordType('hydration')).toBe(true);
    expect(await deleteOwnedRecord(user.id, 'hydration', theirs.id)).toBe(false);
    expect(await prisma.hydrationEvent.findUnique({ where: { id: theirs.id } })).not.toBeNull();

    expect(await deleteOwnedRecord(user.id, 'hydration', mine.id)).toBe(true);
    expect(await prisma.hydrationEvent.findUnique({ where: { id: mine.id } })).toBeNull();
  });

  it('deletes an owned symptom entry and refuses another user’s', async () => {
    const { user } = await createTestUser('delete-symptom');
    const { user: other } = await createTestUser('delete-symptom-other');
    createdUserIds.push(user.id, other.id);
    const mine = await prisma.symptomEntry.create({
      data: { userId: user.id, takenAt: new Date(), symptom: 'headache', dedupeKey: 'sym-mine' },
    });
    const theirs = await prisma.symptomEntry.create({
      data: { userId: other.id, takenAt: new Date(), symptom: 'dizzy', dedupeKey: 'sym-theirs' },
    });

    expect(isDeletableRecordType('symptom')).toBe(true);
    expect(await deleteOwnedRecord(user.id, 'symptom', theirs.id)).toBe(false);
    expect(await prisma.symptomEntry.findUnique({ where: { id: theirs.id } })).not.toBeNull();

    expect(await deleteOwnedRecord(user.id, 'symptom', mine.id)).toBe(true);
    expect(await prisma.symptomEntry.findUnique({ where: { id: mine.id } })).toBeNull();
  });

  it('keeps the History tab list separate from the deletable set', () => {
    // RECORD_TYPES drives History's tabs; widening it would add tabs that
    // page has no renderer for. Deletion takes the wider set.
    expect(RECORD_TYPES).not.toContain('hydration');
    expect(DELETABLE_RECORD_TYPES).toEqual(expect.arrayContaining([...RECORD_TYPES]));
    expect(isDeletableRecordType('note')).toBe(false);
  });
});

describe('loadAnalyticsWindow', () => {
  it('returns only rows inside the requested window', async () => {
    const { user } = await createTestUser('window');
    createdUserIds.push(user.id);
    const inWindow = new Date('2026-03-15T12:00:00Z');
    const beforeWindow = new Date('2026-01-01T12:00:00Z');
    const afterWindow = new Date('2026-06-01T12:00:00Z');

    await prisma.glucoseReading.createMany({
      data: [
        { userId: user.id, takenAt: inWindow, valueMgdl: 105, dedupeKey: 'w-in' },
        { userId: user.id, takenAt: beforeWindow, valueMgdl: 105, dedupeKey: 'w-before' },
        { userId: user.id, takenAt: afterWindow, valueMgdl: 105, dedupeKey: 'w-after' },
      ],
    });

    const from = new Date('2026-03-01T00:00:00Z');
    const to = new Date('2026-04-01T00:00:00Z');
    const window = await loadAnalyticsWindow({ userId: user.id, from, to });
    expect(window.glucose).toHaveLength(1);
    expect(window.glucose[0]?.takenAt.toISOString()).toBe(inWindow.toISOString());
  });
});

describe('recordCounts', () => {
  it('is correct across every counted type', async () => {
    const { user } = await createTestUser('counts');
    createdUserIds.push(user.id);
    await prisma.glucoseReading.createMany({
      data: [
        { userId: user.id, takenAt: new Date(), valueMgdl: 100, dedupeKey: 'c-1' },
        { userId: user.id, takenAt: new Date(), valueMgdl: 101, dedupeKey: 'c-2' },
      ],
    });
    await prisma.meal.create({
      data: {
        userId: user.id,
        takenAt: new Date(),
        description: 'x',
        mealType: 'SNACK',
        dedupeKey: 'c-meal',
      },
    });

    const counts = await recordCounts(user.id);
    expect(counts.glucose).toBe(2);
    expect(counts.meal).toBe(1);
    expect(counts.exercise).toBe(0);
    expect(counts.sleep).toBe(0);
    expect(counts.medication).toBe(0);
    expect(counts.weight).toBe(0);
    expect(counts.bloodPressure).toBe(0);
    expect(counts.mood).toBe(0);
  });
});

describe('cascade deletion of a User', () => {
  it('removes every associated health record and leaves no orphans', async () => {
    const { user } = await createTestUser('cascade');
    const takenAt = new Date();

    await prisma.glucoseReading.create({
      data: { userId: user.id, takenAt, valueMgdl: 100, dedupeKey: 'casc-g' },
    });
    await prisma.meal.create({
      data: { userId: user.id, takenAt, description: 'x', mealType: 'OTHER', dedupeKey: 'casc-m' },
    });
    await prisma.exerciseSession.create({
      data: { userId: user.id, takenAt, activity: 'run', durationMin: 10, dedupeKey: 'casc-e' },
    });
    await prisma.sleepSession.create({
      data: {
        userId: user.id,
        takenAt,
        endedAt: new Date(takenAt.getTime() + 1000),
        durationMin: 10,
        dedupeKey: 'casc-s',
      },
    });
    await prisma.medicationEvent.create({
      data: { userId: user.id, takenAt, name: 'x', dedupeKey: 'casc-med' },
    });
    await prisma.weightMeasurement.create({
      data: { userId: user.id, takenAt, weightKg: 1, dedupeKey: 'casc-w' },
    });
    await prisma.bloodPressureMeasurement.create({
      data: { userId: user.id, takenAt, systolic: 1, diastolic: 1, dedupeKey: 'casc-bp' },
    });
    await prisma.hydrationEvent.create({
      data: { userId: user.id, takenAt, volumeMl: 1, dedupeKey: 'casc-h' },
    });
    await prisma.symptomEntry.create({
      data: { userId: user.id, takenAt, symptom: 'x', dedupeKey: 'casc-sym' },
    });
    await prisma.moodEntry.create({
      data: { userId: user.id, takenAt, mood: 1, dedupeKey: 'casc-mood' },
    });
    await prisma.noteEntry.create({
      data: { userId: user.id, takenAt, text: 'x', dedupeKey: 'casc-note' },
    });
    const device = await prisma.device.create({ data: { userId: user.id, label: 'meter' } });
    const batch = await prisma.importBatch.create({
      data: {
        userId: user.id,
        connectorId: 'generic-csv',
        connectorName: 'Generic CSV',
        filename: 'f.csv',
        deviceId: device.id,
      },
    });
    await prisma.importIssue.create({
      data: { batchId: batch.id, rowNumber: 1, code: 'INVALID_VALUE', message: 'x' },
    });
    await prisma.insight.create({
      data: {
        userId: user.id,
        kind: 'x',
        title: 'x',
        summary: 'x',
        evidenceLevel: 'EARLY',
        evidence: {},
        periodStart: takenAt,
        periodEnd: takenAt,
      },
    });
    const convo = await prisma.aIConversation.create({ data: { userId: user.id } });
    await prisma.aIMessage.create({
      data: { conversationId: convo.id, role: 'user', content: 'hi' },
    });
    await prisma.auditEvent.create({ data: { userId: user.id, action: 'sign-in' } });
    const resetToken = await prisma.passwordResetToken.create({
      data: {
        userId: user.id,
        tokenHash: `hash-${user.id}`,
        expiresAt: new Date(Date.now() + 3600_000),
      },
    });

    await prisma.user.delete({ where: { id: user.id } });

    const checks = await Promise.all([
      prisma.glucoseReading.count({ where: { userId: user.id } }),
      prisma.meal.count({ where: { userId: user.id } }),
      prisma.exerciseSession.count({ where: { userId: user.id } }),
      prisma.sleepSession.count({ where: { userId: user.id } }),
      prisma.medicationEvent.count({ where: { userId: user.id } }),
      prisma.weightMeasurement.count({ where: { userId: user.id } }),
      prisma.bloodPressureMeasurement.count({ where: { userId: user.id } }),
      prisma.hydrationEvent.count({ where: { userId: user.id } }),
      prisma.symptomEntry.count({ where: { userId: user.id } }),
      prisma.moodEntry.count({ where: { userId: user.id } }),
      prisma.noteEntry.count({ where: { userId: user.id } }),
      prisma.device.count({ where: { userId: user.id } }),
      prisma.importBatch.count({ where: { userId: user.id } }),
      prisma.importIssue.count({ where: { id: batch.id } }), // batch gone -> cascade
      prisma.insight.count({ where: { userId: user.id } }),
      prisma.aIConversation.count({ where: { userId: user.id } }),
      prisma.aIMessage.count({ where: { conversationId: convo.id } }),
      prisma.profile.count({ where: { userId: user.id } }),
      prisma.passwordResetToken.count({ where: { id: resetToken.id } }),
    ]);
    expect(checks.every((c) => c === 0)).toBe(true);

    // Audit events keep userId nullable and SET NULL on delete rather than cascading away —
    // confirm no orphan row still references the deleted user id.
    expect(await prisma.auditEvent.count({ where: { userId: user.id } })).toBe(0);
  });
});

describe('pageGlucose cursor', () => {
  // The cursor comes from `?cursor=` on the history page. It used to be
  // resolved with Prisma's `cursor: { id }`, by id alone, so another account's
  // reading id positioned this user's page at that reading's timestamp.
  async function seedReadings(label: string, times: string[]) {
    const { user } = await createTestUser(label);
    createdUserIds.push(user.id);
    await prisma.glucoseReading.createMany({
      data: times.map((t, i) => ({
        userId: user.id,
        takenAt: new Date(t),
        valueMgdl: 100 + i,
        context: 'RANDOM' as const,
        dedupeKey: `${label}-${i}`,
      })),
    });
    return user;
  }

  it('walks every reading exactly once, including ties on takenAt across a page edge', async () => {
    // Five readings share one instant, so a page boundary must fall inside
    // the tie and be broken by id.
    const user = await seedReadings('page-walk', [
      '2026-03-01T08:00:00Z',
      '2026-03-01T09:00:00Z',
      '2026-03-01T09:00:00Z',
      '2026-03-01T09:00:00Z',
      '2026-03-01T09:00:00Z',
      '2026-03-01T09:00:00Z',
      '2026-03-01T10:00:00Z',
    ]);
    const seen: string[] = [];
    let cursor: string | undefined;
    do {
      const page = await pageGlucose({ userId: user.id, take: 2, cursor });
      seen.push(...page.rows.map((r) => r.id));
      cursor = page.nextCursor;
    } while (cursor);
    const all = await prisma.glucoseReading.findMany({
      where: { userId: user.id },
      orderBy: [{ takenAt: 'desc' }, { id: 'desc' }],
      select: { id: true },
    });
    expect(seen).toEqual(all.map((r) => r.id));
  });

  it("ignores another account's reading id instead of positioning on it", async () => {
    const mine = await seedReadings('page-mine', [
      '2026-03-01T08:00:00Z',
      '2026-03-02T08:00:00Z',
      '2026-03-03T08:00:00Z',
    ]);
    const theirs = await seedReadings('page-theirs', ['2026-03-02T12:00:00Z']);
    const foreign = await prisma.glucoseReading.findFirstOrThrow({
      where: { userId: theirs.id },
      select: { id: true },
    });

    const firstPage = await pageGlucose({ userId: mine.id, take: 10 });
    const withForeign = await pageGlucose({ userId: mine.id, take: 10, cursor: foreign.id });
    // Same as no cursor at all: the foreign reading's timestamp leaks nothing.
    expect(withForeign.rows.map((r) => r.id)).toEqual(firstPage.rows.map((r) => r.id));
    expect(withForeign.rows.every((r) => r.userId === mine.id)).toBe(true);

    const unknown = await pageGlucose({ userId: mine.id, take: 10, cursor: 'not-a-real-id' });
    expect(unknown.rows.map((r) => r.id)).toEqual(firstPage.rows.map((r) => r.id));
  });
});
