/**
 * The shared demo account is read-only for credentials and deletion.
 *
 * Its password is published (README, prisma/seed.ts), so anyone can sign in
 * as it. Before this, one visitor could change the password or sign everyone
 * out (locking every later visitor out until a re-seed), link their own
 * Google account as a backdoor, or delete the account or its records. Each
 * path is exercised here through the real handler or Server Action, with only
 * Next's request-scoped APIs (cookies, after, revalidatePath) stood in for.
 */
import { randomUUID } from 'node:crypto';
import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';

const jar = vi.hoisted(() => ({ session: undefined as string | undefined }));
vi.mock('next/headers', () => ({
  cookies: async () => ({
    get: (name: string) =>
      name === 'dialog_session' && jar.session ? { name, value: jar.session } : undefined,
    set: () => undefined,
    delete: () => undefined,
  }),
  headers: async () => new Headers({ 'x-real-ip': randomUUID() }),
}));
vi.mock('next/cache', () => ({ revalidatePath: () => undefined }));
const afterTasks = vi.hoisted(() => [] as Array<() => Promise<void>>);
vi.mock('next/server', async (importOriginal) => ({
  ...(await importOriginal<typeof import('next/server')>()),
  after: (callback: () => Promise<void>) => {
    afterTasks.push(callback);
  },
}));
// The callback's two network calls to Google, replaced so the link branch can
// be reached; everything after them (the part under test) is real.
vi.mock('@/lib/auth/oauth/google', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/lib/auth/oauth/google')>()),
  exchangeCode: vi.fn(async () => ({ idToken: 'stub' })),
  verifyIdToken: vi.fn(async () => ({
    subject: 'demo-link-attempt',
    email: 'visitor@example.test',
    emailVerified: true,
    name: null,
  })),
}));

import { NextRequest } from 'next/server';
import type { User } from '@prisma/client';
import { POST as changePassword } from '@/app/api/auth/password/route';
import { POST as revokeSessions } from '@/app/api/auth/sessions/revoke/route';
import { POST as disconnectGoogle } from '@/app/api/auth/google/disconnect/route';
import { GET as startGoogle } from '@/app/api/auth/google/start/route';
import { GET as googleCallback } from '@/app/api/auth/google/callback/route';
import {
  deleteAccountAction,
  deleteAllRecordsAction,
  updatePreferencesAction,
} from '@/lib/actions/preferences';
import { requestPasswordReset } from '@/lib/auth/password-reset';
import { DEMO_EMAIL, DEMO_READ_ONLY_MESSAGE } from '@/lib/auth/demo';
import { OAUTH_COOKIE, sealAttempt } from '@/lib/auth/oauth/state';
import { pruneRateLimits } from '@/lib/auth/rate-limit';
import { signSession } from '@/lib/auth/session';
import { takeCapturedEmails } from '@/lib/email/providers/console';
import { hashPassword } from '@/lib/auth/password';
import { createTestUser, deleteTestUser, prisma } from './test-helpers';

const ORIGIN = 'http://localhost:3100';
const PASSWORD = 'demo-account-test-password';

let demo: User;
let createdDemo = false;
const otherIds: string[] = [];

beforeAll(async () => {
  // The test database may already hold a seeded demo account; reuse it rather
  // than fight its unique email, and leave it exactly as found.
  const existing = await prisma.user.findUnique({ where: { email: DEMO_EMAIL } });
  if (existing) {
    demo = existing;
  } else {
    demo = await prisma.user.create({
      data: {
        email: DEMO_EMAIL,
        passwordHash: await hashPassword(PASSWORD),
        profile: { create: { onboardingCompletedAt: new Date() } },
      },
    });
    createdDemo = true;
  }
  await prisma.glucoseReading.create({
    data: {
      userId: demo.id,
      takenAt: new Date('2026-01-01T08:00:00Z'),
      valueMgdl: 101,
      dedupeKey: `demo-test-${randomUUID()}`,
    },
  });
});

afterAll(async () => {
  await prisma.authIdentity.deleteMany({
    where: { userId: demo.id, providerAccountId: { startsWith: 'demo-test-' } },
  });
  await prisma.glucoseReading.deleteMany({
    where: { userId: demo.id, dedupeKey: { startsWith: 'demo-test-' } },
  });
  if (createdDemo) await deleteTestUser(demo.id);
  for (const id of otherIds) await deleteTestUser(id);
});

beforeEach(async () => {
  pruneRateLimits(Infinity);
  afterTasks.length = 0;
  demo = await prisma.user.findUniqueOrThrow({ where: { id: demo.id } });
  jar.session = await signSession({ userId: demo.id, tokenVersion: demo.tokenVersion });
});

function post(path: string, fields: Record<string, string> = {}): NextRequest {
  return new NextRequest(`${ORIGIN}${path}`, {
    method: 'POST',
    headers: {
      'content-type': 'application/x-www-form-urlencoded',
      origin: ORIGIN,
      cookie: `dialog_session=${jar.session}`,
    },
    body: new URLSearchParams(fields).toString(),
  });
}

function location(response: Response): URL {
  return new URL(response.headers.get('location')!);
}

async function unchanged(): Promise<void> {
  const now = await prisma.user.findUniqueOrThrow({ where: { id: demo.id } });
  expect(now.passwordHash).toBe(demo.passwordHash);
  expect(now.tokenVersion).toBe(demo.tokenVersion);
}

describe('credential route handlers refuse the demo account', () => {
  it('will not change its password', async () => {
    const response = await changePassword(
      post('/api/auth/password', {
        currentPassword: PASSWORD,
        newPassword: 'a-visitor-chosen-password',
        confirmPassword: 'a-visitor-chosen-password',
      }),
    );
    expect(response.status).toBe(303);
    expect(location(response).pathname).toBe('/app/settings');
    expect(location(response).searchParams.get('error')).toBe('demo_read_only');
    // No replacement cookie: nothing was revoked, so nothing needs reissuing.
    expect(response.headers.get('set-cookie')).toBeNull();
    await unchanged();
  });

  it('will not sign every other visitor out', async () => {
    const response = await revokeSessions(post('/api/auth/sessions/revoke'));
    expect(location(response).searchParams.get('error')).toBe('demo_read_only');
    await unchanged();
  });

  it('will not disconnect a linked identity', async () => {
    const identity = await prisma.authIdentity.create({
      data: { userId: demo.id, provider: 'google', providerAccountId: `demo-test-${randomUUID()}` },
    });
    try {
      const response = await disconnectGoogle(post('/api/auth/google/disconnect'));
      expect(location(response).searchParams.get('error')).toBe('demo_read_only');
      expect(await prisma.authIdentity.findUnique({ where: { id: identity.id } })).not.toBeNull();
    } finally {
      await prisma.authIdentity.delete({ where: { id: identity.id } });
    }
  });

  it('will not start, or complete, linking Google', async () => {
    vi.stubEnv('GOOGLE_CLIENT_ID', 'test-client');
    vi.stubEnv('GOOGLE_CLIENT_SECRET', 'test-secret');
    try {
      const start = await startGoogle(
        new NextRequest(`${ORIGIN}/api/auth/google/start?mode=link`, {
          headers: { 'x-real-ip': randomUUID() },
        }),
      );
      expect(location(start).pathname).toBe('/app/settings');
      expect(location(start).searchParams.get('error')).toBe('demo_read_only');

      // The callback is the binding check: a link attempt begun under some
      // other session must still be refused once it lands on the demo.
      const attempt = await sealAttempt({
        state: 'state-1',
        nonce: 'nonce-1',
        verifier: 'verifier-1',
        mode: 'link',
        next: '/app/settings',
      });
      const callback = await googleCallback(
        new NextRequest(`${ORIGIN}/api/auth/google/callback?code=c&state=state-1`, {
          headers: { cookie: `${OAUTH_COOKIE}=${attempt}` },
        }),
      );
      expect(location(callback).searchParams.get('error')).toBe('demo_read_only');
      expect(
        await prisma.authIdentity.count({
          where: { providerAccountId: 'demo-link-attempt' },
        }),
      ).toBe(0);
    } finally {
      vi.unstubAllEnvs();
    }
  });

  it('still lets an ordinary account start linking', async () => {
    const { user } = await createTestUser('demo-contrast');
    otherIds.push(user.id);
    jar.session = await signSession({ userId: user.id, tokenVersion: user.tokenVersion });
    vi.stubEnv('GOOGLE_CLIENT_ID', 'test-client');
    vi.stubEnv('GOOGLE_CLIENT_SECRET', 'test-secret');
    try {
      const start = await startGoogle(
        new NextRequest(`${ORIGIN}/api/auth/google/start?mode=link`, {
          headers: { 'x-real-ip': randomUUID() },
        }),
      );
      expect(location(start).searchParams.get('error')).toBeNull();
      expect(location(start).origin).not.toBe(ORIGIN);
    } finally {
      vi.unstubAllEnvs();
    }
  });
});

describe('deletion Server Actions refuse the demo account', () => {
  function confirm(): FormData {
    const form = new FormData();
    form.set('confirmEmail', DEMO_EMAIL);
    form.set('password', PASSWORD);
    return form;
  }

  it('will not delete the account', async () => {
    expect(await deleteAccountAction(null, confirm())).toEqual({
      ok: false,
      message: DEMO_READ_ONLY_MESSAGE,
    });
    expect(await prisma.user.findUnique({ where: { id: demo.id } })).not.toBeNull();
  });

  it('will not delete its records', async () => {
    const before = await prisma.glucoseReading.count({ where: { userId: demo.id } });
    expect(await deleteAllRecordsAction(null, confirm())).toEqual({
      ok: false,
      message: DEMO_READ_ONLY_MESSAGE,
    });
    expect(await prisma.glucoseReading.count({ where: { userId: demo.id } })).toBe(before);
  });
});

describe('preferences on the demo account', () => {
  // Consent is personal, the demo is shared: a grant by one visitor would send
  // every later visitor's free text to an external provider.
  async function withProfileRestored(run: () => Promise<void>): Promise<void> {
    const snapshot = await prisma.profile.findUniqueOrThrow({ where: { userId: demo.id } });
    try {
      await prisma.profile.update({
        where: { userId: demo.id },
        data: { externalAiConsentAt: null },
      });
      await run();
    } finally {
      const { id: _id, userId: _userId, ...data } = snapshot;
      await prisma.profile.update({ where: { userId: demo.id }, data });
    }
  }

  function prefs(extra: Record<string, string>): FormData {
    const form = new FormData();
    form.set('glucoseUnit', 'MGDL');
    form.set('locale', 'en-CA');
    form.set('timezone', 'America/Toronto');
    form.set('targetLow', '70');
    form.set('targetHigh', '180');
    form.set('detailLevel', 'STANDARD');
    form.set('aiEnabled', 'on');
    for (const [key, value] of Object.entries(extra)) form.set(key, value);
    return form;
  }

  it('refuses to grant external-AI consent', async () => {
    await withProfileRestored(async () => {
      expect(await updatePreferencesAction(null, prefs({ externalAiConsent: 'on' }))).toEqual({
        ok: false,
        message: DEMO_READ_ONLY_MESSAGE,
      });
      const profile = await prisma.profile.findUniqueOrThrow({ where: { userId: demo.id } });
      expect(profile.externalAiConsentAt).toBeNull();
    });
  });

  it('still saves every other preference', async () => {
    await withProfileRestored(async () => {
      const result = await updatePreferencesAction(null, prefs({ largeText: 'on' }));
      expect(result.ok).toBe(true);
      const profile = await prisma.profile.findUniqueOrThrow({ where: { userId: demo.id } });
      expect(profile.largeText).toBe(true);
      expect(profile.externalAiConsentAt).toBeNull();
    });
  });
});

describe('password recovery skips the demo account', () => {
  it('answers neutrally but issues no reset link', async () => {
    vi.stubEnv('EMAIL_PROVIDER', 'console');
    vi.stubEnv('NEXT_PUBLIC_APP_URL', ORIGIN);
    takeCapturedEmails();
    try {
      const before = await prisma.passwordResetToken.count({ where: { userId: demo.id } });
      const result = await requestPasswordReset({ email: DEMO_EMAIL }, randomUUID());
      expect(result.ok).toBe(true);
      await Promise.all(afterTasks.splice(0).map((task) => task()));
      expect(takeCapturedEmails()).toEqual([]);
      expect(await prisma.passwordResetToken.count({ where: { userId: demo.id } })).toBe(before);
    } finally {
      vi.unstubAllEnvs();
    }
  });
});
