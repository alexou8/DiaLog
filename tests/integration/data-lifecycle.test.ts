/**
 * Deletion and audit-trail guarantees.
 *
 * These cover three defects found in the September 2026 audit, each of which
 * was a privacy or forensics problem rather than a crash:
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
 */
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
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
