/**
 * Deletion and audit-trail guarantees.
 *
 * These cover defects found in the September and October 2026 audits, each of
 * which was a privacy or forensics problem rather than a crash:
 *
 *   1. "Delete all my records" wiped the eleven health-record tables but left
 *      AI conversations (free-text health discussion plus the evidence behind
 *      each answer) and import batches (whose ImportIssue.rawRow holds raw
 *      rejected health rows) in place.
 *   2. AuditEvent.userId was ON DELETE CASCADE, so deleting an account also
 *      deleted the `auth.account_delete` event recording the deletion — the
 *      audit trail erased itself precisely when it mattered.
 *   3. An AIConversation could be addressed by id alone, letting one user
 *      append messages to another user's conversation (see assistant-*).
 *   4. "Delete all my records" also left Insight rows (statements about the
 *      user's readings) and Device rows (which meter or CGM they use).
 */
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest';

// deleteAllRecordsAction is a Server Action: it reads the session through
// next/headers and revalidates through next/cache, neither of which exists
// outside a Next request. Only those two request-scoped APIs are stood in for,
// so the tests below drive the real action (authentication, confirmation, the
// deletion itself) instead of re-running copies of its queries.
const jar = vi.hoisted(() => ({ session: undefined as string | undefined }));
vi.mock('next/headers', () => ({
  cookies: async () => ({
    get: (name: string) =>
      name === 'dialog_session' && jar.session ? { name, value: jar.session } : undefined,
    set: () => undefined,
    delete: () => undefined,
  }),
  headers: async () => new Headers(),
}));
vi.mock('next/cache', () => ({ revalidatePath: () => undefined }));

import { deleteAllRecordsAction } from '@/lib/actions/preferences';
import { signSession } from '@/lib/auth/session';
import { prisma, createTestUser, deleteTestUser, type SeededUser } from './test-helpers';

let user: SeededUser;

beforeAll(async () => {
  user = await createTestUser('lifecycle');
});

afterAll(async () => {
  await deleteTestUser(user.user.id);
});

describe('AuditEvent survives the account it names', () => {
  it('keeps the row with a null userId when the user is deleted', async () => {
    const doomed = await createTestUser('lifecycle-doomed');
    const event = await prisma.auditEvent.create({
      data: { userId: doomed.user.id, action: 'auth.account_delete' },
    });

    await prisma.user.delete({ where: { id: doomed.user.id } });

    const after = await prisma.auditEvent.findUnique({ where: { id: event.id } });
    // Under the old ON DELETE CASCADE this was null: the event vanished with
    // the account. The action must remain on the record, de-identified.
    expect(after).not.toBeNull();
    expect(after?.userId).toBeNull();
    expect(after?.action).toBe('auth.account_delete');

    await prisma.auditEvent.delete({ where: { id: event.id } });
  });
});

describe('derived health data is reachable for deletion', () => {
  it('cascades AI messages and import issues from the parents delete-all now clears', async () => {
    const conversation = await prisma.aIConversation.create({
      data: { userId: user.user.id, title: 'what raises my morning readings?' },
    });
    await prisma.aIMessage.create({
      data: { conversationId: conversation.id, role: 'user', content: 'free-text health question' },
    });
    const batch = await prisma.importBatch.create({
      data: {
        userId: user.user.id,
        connectorId: 'generic-csv',
        connectorName: 'Generic CSV',
        filename: 'export.csv',
        status: 'COMPLETED',
      },
    });
    await prisma.importIssue.create({
      data: {
        batchId: batch.id,
        rowNumber: 1,
        code: 'BAD_ROW',
        message: 'x',
        rawRow: '2026-01-01,142',
      },
    });

    // deleteAllRecordsAction deletes these two parents; both children must go
    // with them, or raw health rows survive a "delete everything" request.
    await prisma.$transaction([
      prisma.aIConversation.deleteMany({ where: { userId: user.user.id } }),
      prisma.importBatch.deleteMany({ where: { userId: user.user.id } }),
    ]);

    expect(await prisma.aIMessage.count({ where: { conversationId: conversation.id } })).toBe(0);
    expect(await prisma.importIssue.count({ where: { batchId: batch.id } })).toBe(0);
  });
});

describe('AIConversation ownership', () => {
  it("does not resolve another user's conversation by id alone", async () => {
    const other = await createTestUser('lifecycle-other');
    const victim = await prisma.aIConversation.create({
      data: { userId: other.user.id, title: 'private' },
    });

    // This is the lookup askAssistantAction now performs. Scoped by owner, a
    // guessed or leaked id from another account simply does not match, so the
    // action creates a fresh conversation instead of writing into the victim's.
    const resolved = await prisma.aIConversation.findFirst({
      where: { id: victim.id, userId: user.user.id },
      select: { id: true },
    });
    expect(resolved).toBeNull();

    await deleteTestUser(other.user.id);
  });
});

describe('deleteAllRecordsAction, called for real', () => {
  const PASSWORD = 'a-reasonably-long-test-password-1';

  async function seedEverything(userId: string) {
    const takenAt = new Date('2026-02-01T08:00:00Z');
    const key = () => `lifecycle-${Math.random().toString(36).slice(2)}`;
    const device = await prisma.device.create({
      data: { userId, label: 'Kitchen meter', vendor: 'Example', model: 'G1' },
    });
    await prisma.glucoseReading.create({
      data: { userId, takenAt, valueMgdl: 123, deviceId: device.id, dedupeKey: key() },
    });
    await prisma.meal.create({
      data: {
        userId,
        takenAt,
        description: 'toast',
        mealType: 'BREAKFAST',
        dedupeKey: key(),
        foodItems: { create: { name: 'toast' } },
      },
    });
    await prisma.noteEntry.create({ data: { userId, takenAt, text: 'private', dedupeKey: key() } });
    await prisma.moodEntry.create({ data: { userId, takenAt, mood: 3, dedupeKey: key() } });
    await prisma.insight.create({
      data: {
        userId,
        kind: 'post_meal',
        title: 'Higher after dinner',
        summary: 'Your post-dinner readings run higher than after lunch.',
        evidenceLevel: 'EMERGING',
        evidence: { readings: 12 },
        periodStart: takenAt,
        periodEnd: takenAt,
      },
    });
    const conversation = await prisma.aIConversation.create({ data: { userId, title: 'q' } });
    await prisma.aIMessage.create({
      data: { conversationId: conversation.id, role: 'user', content: 'free text' },
    });
    const batch = await prisma.importBatch.create({
      data: {
        userId,
        connectorId: 'generic-csv',
        connectorName: 'Generic CSV',
        filename: 'export.csv',
        status: 'COMPLETED',
      },
    });
    await prisma.importIssue.create({
      data: { batchId: batch.id, rowNumber: 1, code: 'BAD_ROW', message: 'x', rawRow: '1,2' },
    });
  }

  async function remaining(userId: string): Promise<Record<string, number>> {
    const where = { userId };
    return {
      glucose: await prisma.glucoseReading.count({ where }),
      meal: await prisma.meal.count({ where }),
      foodItem: await prisma.foodItem.count({ where: { meal: { userId } } }),
      note: await prisma.noteEntry.count({ where }),
      mood: await prisma.moodEntry.count({ where }),
      insight: await prisma.insight.count({ where }),
      device: await prisma.device.count({ where }),
      conversation: await prisma.aIConversation.count({ where }),
      message: await prisma.aIMessage.count({ where: { conversation: { userId } } }),
      batch: await prisma.importBatch.count({ where }),
      issue: await prisma.importIssue.count({ where: { batch: { userId } } }),
    };
  }

  it('erases every health-derived model for the signed-in user only', async () => {
    const target = await createTestUser('lifecycle-action', { password: PASSWORD });
    const bystander = await createTestUser('lifecycle-bystander');
    try {
      await seedEverything(target.user.id);
      await seedEverything(bystander.user.id);
      jar.session = await signSession({
        userId: target.user.id,
        tokenVersion: target.user.tokenVersion,
      });

      const form = new FormData();
      form.set('confirmEmail', target.user.email);
      form.set('password', PASSWORD);
      const result = await deleteAllRecordsAction(null, form);

      expect(result.ok).toBe(true);
      // 4 records + insight + conversation + batch. The device is erased too,
      // but it is equipment rather than a record, so it is not counted.
      expect(result.message).toMatch(/^7 records have been permanently deleted/);
      for (const [model, count] of Object.entries(await remaining(target.user.id))) {
        expect(count, model).toBe(0);
      }
      // The account itself survives: this is not account deletion.
      expect(await prisma.profile.count({ where: { userId: target.user.id } })).toBe(1);
      // And nothing of anyone else's was touched.
      for (const [model, count] of Object.entries(await remaining(bystander.user.id))) {
        expect(count, model).toBeGreaterThan(0);
      }
    } finally {
      jar.session = undefined;
      await deleteTestUser(target.user.id);
      await deleteTestUser(bystander.user.id);
    }
  });

  it('deletes nothing when the confirmation is wrong', async () => {
    const target = await createTestUser('lifecycle-wrong', { password: PASSWORD });
    try {
      await seedEverything(target.user.id);
      jar.session = await signSession({
        userId: target.user.id,
        tokenVersion: target.user.tokenVersion,
      });
      const form = new FormData();
      form.set('confirmEmail', target.user.email);
      form.set('password', 'not-the-password');
      const result = await deleteAllRecordsAction(null, form);
      expect(result.ok).toBe(false);
      expect((await remaining(target.user.id)).insight).toBe(1);
    } finally {
      jar.session = undefined;
      await deleteTestUser(target.user.id);
    }
  });
});
