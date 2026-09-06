import { randomUUID } from 'node:crypto';

// Hand-built requests have no Next request lifecycle. Capture after() callbacks
// and drain them explicitly; the browser suite exercises Next's real scheduler.
const afterTasks = vi.hoisted(() => [] as Array<() => Promise<void>>);
vi.mock('next/server', async (importOriginal) => ({
  ...(await importOriginal<typeof import('next/server')>()),
  after: (callback: () => Promise<void>) => {
    afterTasks.push(callback);
  },
}));
async function finishAfter() {
  await Promise.all(afterTasks.splice(0).map((callback) => callback()));
}

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const db = vi.hoisted(() => ({
  user: { findUnique: vi.fn() },
  passwordResetToken: { updateMany: vi.fn(), create: vi.fn() },
  $queryRaw: vi.fn(),
  $transaction: vi.fn(),
}));
const mail = vi.hoisted(() => ({ available: vi.fn(), sendRecovery: vi.fn() }));
const audit = vi.hoisted(() => vi.fn());
vi.mock('@/lib/db/prisma', () => ({ prisma: db }));
vi.mock('@/lib/auth/audit', () => ({ audit }));
vi.mock('@/lib/email/provider', () => ({ getEmailProvider: () => mail }));
import {
  requestPasswordReset,
  RESET_CONFIRMATION,
  RECOVERY_UNAVAILABLE,
} from '@/lib/auth/password-reset';
import { RATE_LIMITS, pruneRateLimits } from '@/lib/auth/rate-limit';

beforeEach(() => {
  afterTasks.length = 0;
  vi.resetAllMocks();
  vi.stubEnv('AUTH_SECRET', 'unit-recovery-secret-at-least-32-characters');
  vi.stubEnv('NEXT_PUBLIC_APP_URL', 'https://dialog.example');
  mail.available.mockReturnValue(true);
  db.$transaction.mockImplementation((fn) => fn(db));
  pruneRateLimits(Infinity);
});
afterEach(() => {
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

describe('recovery request neutrality', () => {
  it.each([
    null,
    { id: 'google-owner', passwordHash: null },
    { id: 'password-owner', passwordHash: 'bcrypt-hash' },
  ])('returns identical confirmation for every account state', async (user) => {
    db.user.findUnique.mockResolvedValue(user);
    const state = await requestPasswordReset({ email: 'person@example.com' }, randomUUID());
    expect(state).toEqual({ ok: true, message: RESET_CONFIRMATION });
    expect(mail.sendRecovery).not.toHaveBeenCalled();
    await finishAfter();
    expect(mail.sendRecovery).toHaveBeenCalledTimes(user?.passwordHash ? 1 : 0);
    if (user?.passwordHash) {
      const message = mail.sendRecovery.mock.calls[0]![0];
      const raw = new URL(message.url).searchParams.get('token');
      expect(JSON.stringify(db.passwordResetToken.create.mock.calls)).not.toContain(raw);
      expect(JSON.stringify(audit.mock.calls)).not.toContain(raw);
      expect(JSON.stringify(state)).not.toContain(raw);
      expect(new URL(message.url).origin).toBe('https://dialog.example');
    }
  });
  it('returns neutral confirmation without awaiting a mail send that never settles', async () => {
    db.user.findUnique.mockResolvedValue({ id: 'owner', passwordHash: 'hash' });
    mail.sendRecovery.mockImplementation(() => new Promise<void>(() => {}));
    const state = await requestPasswordReset({ email: 'person@example.com' }, randomUUID());
    expect(state).toEqual({ ok: true, message: RESET_CONFIRMATION });
    expect(afterTasks).toHaveLength(1);
    expect(db.$transaction).not.toHaveBeenCalled();
    expect(mail.sendRecovery).not.toHaveBeenCalled();
    // The request has resolved before Next starts the captured callback. No
    // floating production promise or timing threshold is needed to prove it.
  });
  it('validates email without consulting the database', async () => {
    expect(await requestPasswordReset({ email: 'bad' }, randomUUID())).toMatchObject({
      ok: false,
      errors: { email: expect.any(String) },
    });
    expect(db.user.findUnique).not.toHaveBeenCalled();
  });
  it('throttles the normalized email across different IPs without revealing the account', async () => {
    db.user.findUnique.mockResolvedValue(null);
    for (let i = 0; i <= RATE_LIMITS.passwordResetEmail.limit; i++) {
      expect(
        await requestPasswordReset(
          { email: i % 2 ? ' PERSON@example.com ' : 'person@example.com' },
          randomUUID(),
        ),
      ).toEqual({ ok: true, message: RESET_CONFIRMATION });
    }
    expect(db.user.findUnique).toHaveBeenCalledTimes(RATE_LIMITS.passwordResetEmail.limit);
  });
  it('limits IPs even when each request uses a different address', async () => {
    db.user.findUnique.mockResolvedValue(null);
    const ip = randomUUID();
    for (let i = 0; i < RATE_LIMITS.passwordResetIp.limit; i++)
      await requestPasswordReset({ email: `person${i}@example.com` }, ip);
    expect(await requestPasswordReset({ email: 'last@example.com' }, ip)).toMatchObject({
      ok: false,
    });
    expect(db.user.findUnique).toHaveBeenCalledTimes(RATE_LIMITS.passwordResetIp.limit);
  });
  it.each(['secret', 'transport', 'origin'])(
    'reports configuration failure before account lookup: %s',
    async (setting) => {
      if (setting === 'secret') vi.stubEnv('AUTH_SECRET', '');
      if (setting === 'transport') mail.available.mockReturnValue(false);
      if (setting === 'origin') vi.stubEnv('NEXT_PUBLIC_APP_URL', 'http://untrusted.example');
      expect(await requestPasswordReset({ email: 'person@example.com' }, randomUUID())).toEqual({
        ok: false,
        message: RECOVERY_UNAVAILABLE,
      });
      expect(db.user.findUnique).not.toHaveBeenCalled();
      expect(afterTasks).toHaveLength(0);
    },
  );
  it('invalidates an undelivered token, emits only a fixed operator signal, and stays neutral', async () => {
    db.user.findUnique.mockResolvedValue({ id: 'owner', passwordHash: 'hash' });
    mail.sendRecovery.mockRejectedValue(new Error('sensitive transport body'));
    const log = vi.spyOn(console, 'error').mockImplementation(() => {});
    expect(await requestPasswordReset({ email: 'person@example.com' }, randomUUID())).toEqual({
      ok: true,
      message: RESET_CONFIRMATION,
    });
    await finishAfter();
    expect(db.passwordResetToken.updateMany).toHaveBeenLastCalledWith({
      where: { userId: 'owner', tokenHash: expect.any(String), usedAt: null },
      data: { usedAt: expect.any(Date) },
    });
    expect(audit).toHaveBeenCalledWith({ userId: 'owner', action: 'auth.password_reset_failed' });
    expect(log).toHaveBeenCalledExactlyOnceWith(
      'Password recovery request failed. Check database and email delivery.',
    );
  });
});
