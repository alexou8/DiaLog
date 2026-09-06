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

import { afterAll, afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { NextRequest } from 'next/server';
import { GET, POST } from '@/app/api/auth/password/reset/route';
import {
  requestPasswordReset,
  prepareResetProof,
  RESET_CONFIRMATION,
  RECOVERY_UNAVAILABLE,
} from '@/lib/auth/password-reset';
import { createResetToken, hashResetToken, signResetProof } from '@/lib/auth/password-reset-token';
import { takeCapturedEmails } from '@/lib/email/providers/console';
import { verifyPassword } from '@/lib/auth/password';
import { signSession, verifySession } from '@/lib/auth/session';
import { RATE_LIMITS, pruneRateLimits } from '@/lib/auth/rate-limit';
import { createTestUser, deleteTestUser, prisma } from './test-helpers';

const ORIGIN = 'http://localhost:3100';
const NEXT_PASSWORD = 'new-recovery-password-2026';
const ids: string[] = [];

beforeEach(() => {
  afterTasks.length = 0;
  vi.stubEnv('EMAIL_PROVIDER', 'console');
  vi.stubEnv('NEXT_PUBLIC_APP_URL', ORIGIN);
  pruneRateLimits(Infinity);
  takeCapturedEmails();
});
afterEach(async () => {
  await finishAfter();
  vi.unstubAllEnvs();
  takeCapturedEmails();
});
afterAll(async () => {
  for (const id of ids) await deleteTestUser(id);
});

async function seed(label = 'recovery') {
  const { user } = await createTestUser(label);
  ids.push(user.id);
  return user;
}
function post(
  fields: Record<string, string> = {},
  options: { ip?: string; origin?: string | null; cookie?: string } = {},
) {
  const headers = new Headers({
    'content-type': 'application/x-www-form-urlencoded',
    'x-real-ip': options.ip ?? randomUUID(),
  });
  if (options.origin !== null) headers.set('origin', options.origin ?? ORIGIN);
  if (options.cookie) headers.set('cookie', `dialog_session=${options.cookie}`);
  return new NextRequest(`${ORIGIN}/api/auth/password/reset`, {
    method: 'POST',
    headers,
    body: new URLSearchParams({
      newPassword: NEXT_PASSWORD,
      confirmPassword: NEXT_PASSWORD,
      ...fields,
    }).toString(),
  });
}
function outcome(response: Response) {
  return new URL(response.headers.get('location')!).searchParams.get('password');
}
async function issue(email: string) {
  expect(await requestPasswordReset({ email }, randomUUID())).toEqual({
    ok: true,
    message: RESET_CONFIRMATION,
  });
  await finishAfter();
  const [mail] = takeCapturedEmails();
  if (!mail) throw new Error('Expected captured recovery mail.');
  const raw = new URL(mail.url).searchParams.get('token')!;
  const proof = await prepareResetProof(raw);
  expect(proof).toBeTruthy();
  return { raw, proof: proof!, url: mail.url };
}

describe('password recovery against Postgres', () => {
  it('resets the password, consumes the digest, revokes existing sessions and never signs in', async () => {
    const user = await seed();
    const cookie = await signSession({ userId: user.id, tokenVersion: user.tokenVersion });
    const { raw, proof, url } = await issue(user.email);
    const stored = await prisma.passwordResetToken.findUniqueOrThrow({
      where: { tokenHash: hashResetToken(raw) },
    });
    expect(stored.userId).toBe(user.id);
    const landing = await GET(new NextRequest(url));
    expect(landing.status).toBe(303);
    expect(landing.headers.get('location')).not.toContain(raw);
    expect(await landing.text()).not.toContain(raw);
    expect(landing.headers.get('referrer-policy')).toBe('no-referrer');
    const response = await POST(post({ proof }, { cookie }));
    expect(response.status).toBe(303);
    expect(response.headers.get('location')).toBe(`${ORIGIN}/sign-in?password=reset`);
    expect(response.headers.get('set-cookie')).toContain('Max-Age=0');
    const after = await prisma.user.findUniqueOrThrow({ where: { id: user.id } });
    expect(await verifyPassword(NEXT_PASSWORD, after.passwordHash!)).toBe(true);
    expect(after.tokenVersion).toBe(user.tokenVersion + 1);
    expect((await verifySession(cookie))?.tokenVersion).not.toBe(after.tokenVersion);
    expect(await prepareResetProof(raw)).toBeNull();
    expect(
      (
        await prisma.passwordResetToken.findFirstOrThrow({
          where: { id: stored.id, userId: user.id },
        })
      ).usedAt,
    ).not.toBeNull();
    const events = await prisma.auditEvent.findMany({
      where: { userId: user.id, action: { startsWith: 'auth.password_reset_' } },
    });
    expect(events.map((event) => event.action)).toContain('auth.password_reset_completed');
    expect(JSON.stringify(events)).not.toContain(raw);
  });
  it.each(['expired', 'used'])('refuses a %s token without changing credentials', async (kind) => {
    const user = await seed(kind);
    const { raw, proof } = await issue(user.email);
    await prisma.passwordResetToken.updateMany({
      where: { userId: user.id, tokenHash: hashResetToken(raw) },
      data: kind === 'expired' ? { expiresAt: new Date(Date.now() - 1) } : { usedAt: new Date() },
    });
    expect(await prepareResetProof(raw)).toBeNull();
    expect(outcome(await POST(post({ proof })))).toBe('invalid');
    expect((await prisma.user.findUniqueOrThrow({ where: { id: user.id } })).passwordHash).toBe(
      user.passwordHash,
    );
  });
  it('refuses reuse after success', async () => {
    const user = await seed('reuse');
    const { proof } = await issue(user.email);
    expect(outcome(await POST(post({ proof })))).toBe('reset');
    expect(outcome(await POST(post({ proof })))).toBe('invalid');
    expect((await prisma.user.findUniqueOrThrow({ where: { id: user.id } })).tokenVersion).toBe(1);
  });
  it.each(['', 'malformed', 'a'.repeat(43)])(
    'rejects missing, malformed and unknown credentials',
    async (raw) => {
      expect(await prepareResetProof(raw)).toBeNull();
      expect(outcome(await POST(post({ proof: raw })))).toBe('invalid');
      expect(
        outcome(await GET(new NextRequest(`${ORIGIN}/api/auth/password/reset?token=${raw}`))),
      ).toBe('invalid');
    },
  );
  it('cannot combine another owner with a valid token hash', async () => {
    const owner = await seed('owner');
    const other = await seed('other');
    const { raw, proof } = await issue(owner.email);
    // Even a server-signed mismatched fixture must fail the owner-scoped query.
    const mismatched = await signResetProof(
      { userId: other.id, tokenHash: hashResetToken(raw) },
      new Date(Date.now() + 60_000),
    );
    expect(outcome(await POST(post({ proof: mismatched })))).toBe('invalid');
    // Client-supplied account ids never override the proven recovery owner.
    expect(outcome(await POST(post({ proof, userId: other.id })))).toBe('reset');
    const untouched = await prisma.user.findUniqueOrThrow({ where: { id: other.id } });
    expect(untouched.passwordHash).toBe(other.passwordHash);
    expect(untouched.tokenVersion).toBe(other.tokenVersion);
  });
  it('returns the same confirmation for unknown and Google-only addresses, without creating tokens', async () => {
    const user = await seed('google');
    await prisma.user.update({ where: { id: user.id }, data: { passwordHash: null } });
    for (const email of [user.email, `${randomUUID()}@dialog.test`])
      expect(await requestPasswordReset({ email }, randomUUID())).toEqual({
        ok: true,
        message: RESET_CONFIRMATION,
      });
    await finishAfter();
    expect(takeCapturedEmails()).toEqual([]);
    expect(await prisma.passwordResetToken.count({ where: { userId: user.id } })).toBe(0);
    // A stray legacy token must not turn Google-only recovery into password setup.
    const token = createResetToken();
    await prisma.passwordResetToken.create({
      data: { userId: user.id, tokenHash: token.tokenHash, expiresAt: token.expiresAt },
    });
    const proof = await signResetProof(
      { userId: user.id, tokenHash: token.tokenHash },
      token.expiresAt,
    );
    expect(await prepareResetProof(token.raw)).toBeNull();
    expect(outcome(await POST(post({ proof })))).toBe('invalid');
  });
  it.each([
    ['short', 'short', 'too_short'],
    ['password123', 'password123', 'too_common'],
    ['x'.repeat(201), 'x'.repeat(201), 'too_long'],
    [NEXT_PASSWORD, 'different-password', 'mismatch'],
  ])(
    'rejects password policy/mismatch without consuming the link',
    async (newPassword, confirmPassword, expected) => {
      const user = await seed('policy');
      const { proof, raw } = await issue(user.email);
      const response = await POST(post({ proof, newPassword, confirmPassword }));
      expect(outcome(response)).toBe(expected);
      expect(response.headers.get('location')).not.toContain(raw);
      expect(await prepareResetProof(raw)).toBeTruthy();
      expect((await prisma.user.findUniqueOrThrow({ where: { id: user.id } })).passwordHash).toBe(
        user.passwordHash,
      );
    },
  );
  it.each(['https://attacker.example', null])(
    'rejects foreign or missing Origin',
    async (origin) => {
      expect((await POST(post({}, { origin }))).status).toBe(403);
    },
  );
  it('permits exactly one concurrent submit and one version increment', async () => {
    const user = await seed('concurrent');
    const { proof } = await issue(user.email);
    const responses = await Promise.all([POST(post({ proof })), POST(post({ proof }))]);
    expect(responses.map(outcome).sort()).toEqual(['invalid', 'reset']);
    expect((await prisma.user.findUniqueOrThrow({ where: { id: user.id } })).tokenVersion).toBe(
      user.tokenVersion + 1,
    );
  });
  it('serializes concurrent requests so only the newest issued link remains usable', async () => {
    const user = await seed('issue-race');
    await Promise.all([
      requestPasswordReset({ email: user.email }, randomUUID()),
      requestPasswordReset({ email: user.email }, randomUUID()),
    ]);
    await finishAfter();
    const mail = takeCapturedEmails();
    expect(mail).toHaveLength(2);
    const proofs = await Promise.all(
      mail.map((m) => prepareResetProof(new URL(m.url).searchParams.get('token'))),
    );
    expect(proofs.filter(Boolean)).toHaveLength(1);
    expect(
      await prisma.passwordResetToken.count({ where: { userId: user.id, usedAt: null } }),
    ).toBe(1);
  });
  it('invalidates the prior link when another one is requested', async () => {
    const user = await seed('replace');
    const old = await issue(user.email);
    await issue(user.email);
    expect(await prepareResetProof(old.raw)).toBeNull();
    expect(outcome(await POST(post({ proof: old.proof })))).toBe('invalid');
  });
  it('limits request email and IP independently, and limits submissions', async () => {
    const user = await seed('limits');
    for (let i = 0; i <= RATE_LIMITS.passwordResetEmail.limit; i++)
      await requestPasswordReset({ email: user.email }, randomUUID());
    await finishAfter();
    expect(takeCapturedEmails()).toHaveLength(RATE_LIMITS.passwordResetEmail.limit);
    const ip = randomUUID();
    for (let i = 0; i < RATE_LIMITS.passwordResetIp.limit; i++)
      await requestPasswordReset({ email: `${randomUUID()}@dialog.test` }, ip);
    expect(await requestPasswordReset({ email: user.email }, ip)).toMatchObject({ ok: false });
    for (let i = 0; i < RATE_LIMITS.passwordResetSubmit.limit; i++) await POST(post({}, { ip }));
    expect(outcome(await POST(post({}, { ip })))).toBe('rate_limited');
  });
  it('fails closed before issuing or consuming when AUTH_SECRET is missing', async () => {
    const user = await seed('config');
    const { proof, raw } = await issue(user.email);
    vi.stubEnv('AUTH_SECRET', '');
    expect(await requestPasswordReset({ email: user.email }, randomUUID())).toEqual({
      ok: false,
      message: RECOVERY_UNAVAILABLE,
    });
    expect(outcome(await POST(post({ proof })))).toBe('unavailable');
    expect(await prepareResetProof(raw)).toBeNull();
    expect((await prisma.user.findUniqueOrThrow({ where: { id: user.id } })).passwordHash).toBe(
      user.passwordHash,
    );
  });
  it('cascades reset tokens when the account is deleted', async () => {
    const user = await seed('delete');
    await issue(user.email);
    await deleteTestUser(user.id);
    expect(await prisma.passwordResetToken.count({ where: { userId: user.id } })).toBe(0);
  });
});
