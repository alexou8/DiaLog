/**
 * The two credential-mutating route handlers, against a real database.
 *
 * These exist as route handlers rather than Server Actions because both revoke
 * the session cookie their own request arrived with — see lib/auth/route-form.ts.
 * The behaviour that matters, and that these tests pin down, is the pair:
 * every other device's cookie must stop working, and *this* request's response
 * must carry a replacement so the person doing it stays signed in.
 */
import { afterAll, describe, expect, it } from 'vitest';
import { NextRequest } from 'next/server';
import { SignJWT, jwtVerify } from 'jose';
import { POST as changePassword } from '@/app/api/auth/password/route';
import { POST as revokeSessions } from '@/app/api/auth/sessions/revoke/route';
import { verifyPassword } from '@/lib/auth/password';
import { prisma, createTestUser, deleteTestUser } from './test-helpers';

const ORIGIN = 'http://localhost:3100';
const SESSION_COOKIE = 'dialog_session';
const CURRENT = 'a-reasonably-long-test-password-1';

const createdUserIds: string[] = [];

afterAll(async () => {
  for (const id of createdUserIds) await deleteTestUser(id);
});

function secret(): Uint8Array {
  return new TextEncoder().encode(process.env.AUTH_SECRET as string);
}

/** Mints a cookie the handlers will accept, without going through next/headers. */
async function token(userId: string, tokenVersion: number): Promise<string> {
  return new SignJWT({ tokenVersion })
    .setProtectedHeader({ alg: 'HS256' })
    .setSubject(userId)
    .setIssuer('dialog')
    .setIssuedAt()
    .setExpirationTime('30d')
    .sign(secret());
}

async function readToken(value: string): Promise<{ sub: string; tokenVersion: number }> {
  const { payload } = await jwtVerify(value, secret(), { issuer: 'dialog' });
  return { sub: payload.sub as string, tokenVersion: payload.tokenVersion as number };
}

interface PostOptions {
  cookie?: string;
  origin?: string | null;
  fields?: Record<string, string>;
}

function post(path: string, { cookie, origin = ORIGIN, fields = {} }: PostOptions): NextRequest {
  const headers = new Headers({ 'content-type': 'application/x-www-form-urlencoded' });
  if (cookie) headers.set('cookie', `${SESSION_COOKIE}=${cookie}`);
  if (origin) headers.set('origin', origin);
  return new NextRequest(`${ORIGIN}${path}`, {
    method: 'POST',
    headers,
    body: new URLSearchParams(fields).toString(),
  });
}

/** The `?password=` / `?sessions=` code a 303 carries back to Settings. */
function outcome(response: Response, key: string): string | null {
  const location = response.headers.get('location');
  if (!location) return null;
  return new URL(location).searchParams.get(key);
}

function setCookie(response: Response): string | null {
  const header = response.headers.get('set-cookie');
  if (!header) return null;
  const match = /dialog_session=([^;]+)/.exec(header);
  return match?.[1] ?? null;
}

async function seed(label: string) {
  const { user } = await createTestUser(label, { password: CURRENT });
  createdUserIds.push(user.id);
  return user;
}

describe('POST /api/auth/password', () => {
  it('changes the password, revokes every other cookie, and keeps this one alive', async () => {
    const user = await seed('pw-change');
    const before = await token(user.id, user.tokenVersion);

    const response = await changePassword(
      post('/api/auth/password', {
        cookie: before,
        fields: {
          currentPassword: CURRENT,
          newPassword: 'a-brand-new-password-2026',
          confirmPassword: 'a-brand-new-password-2026',
        },
      }),
    );

    expect(response.status).toBe(303);
    expect(outcome(response, 'password')).toBe('changed');

    const after = await prisma.user.findUniqueOrThrow({ where: { id: user.id } });
    expect(await verifyPassword('a-brand-new-password-2026', after.passwordHash as string)).toBe(
      true,
    );

    // Sign out everywhere: the version moved, so `before` is now worthless.
    expect(after.tokenVersion).toBe(user.tokenVersion + 1);
    expect((await readToken(before)).tokenVersion).not.toBe(after.tokenVersion);

    // ...but the response carries a replacement, so this browser stays in.
    // This is the half that the Server Action could not deliver reliably.
    const reissued = setCookie(response);
    expect(reissued).toBeTruthy();
    const payload = await readToken(reissued as string);
    expect(payload.sub).toBe(user.id);
    expect(payload.tokenVersion).toBe(after.tokenVersion);
  });

  it('refuses a wrong current password without touching the account', async () => {
    const user = await seed('pw-wrong');
    const response = await changePassword(
      post('/api/auth/password', {
        cookie: await token(user.id, user.tokenVersion),
        fields: {
          currentPassword: 'not-the-current-password',
          newPassword: 'a-brand-new-password-2026',
          confirmPassword: 'a-brand-new-password-2026',
        },
      }),
    );

    expect(outcome(response, 'password')).toBe('wrong_current');
    expect(setCookie(response)).toBeNull();

    const after = await prisma.user.findUniqueOrThrow({ where: { id: user.id } });
    expect(after.tokenVersion).toBe(user.tokenVersion);
    expect(await verifyPassword(CURRENT, after.passwordHash as string)).toBe(true);
  });

  it.each([
    [
      'missing_current',
      {
        currentPassword: '',
        newPassword: 'long-enough-password',
        confirmPassword: 'long-enough-password',
      },
    ],
    ['too_short', { currentPassword: CURRENT, newPassword: 'short', confirmPassword: 'short' }],
    [
      'too_common',
      { currentPassword: CURRENT, newPassword: 'password123', confirmPassword: 'password123' },
    ],
    [
      'mismatch',
      {
        currentPassword: CURRENT,
        newPassword: 'long-enough-password',
        confirmPassword: 'a-different-one-entirely',
      },
    ],
  ])('reports %s and leaves the password alone', async (expected, fields) => {
    const user = await seed(`pw-${expected}`);
    const response = await changePassword(
      post('/api/auth/password', { cookie: await token(user.id, user.tokenVersion), fields }),
    );

    expect(outcome(response, 'password')).toBe(expected);
    const after = await prisma.user.findUniqueOrThrow({ where: { id: user.id } });
    expect(after.tokenVersion).toBe(user.tokenVersion);
    expect(await verifyPassword(CURRENT, after.passwordHash as string)).toBe(true);
  });

  it('lets a Google-only account set a first password without signing itself out', async () => {
    const user = await seed('pw-first');
    await prisma.user.update({ where: { id: user.id }, data: { passwordHash: null } });

    const response = await changePassword(
      post('/api/auth/password', {
        cookie: await token(user.id, user.tokenVersion),
        fields: {
          newPassword: 'my-first-real-password',
          confirmPassword: 'my-first-real-password',
        },
      }),
    );

    expect(outcome(response, 'password')).toBe('set');
    const after = await prisma.user.findUniqueOrThrow({ where: { id: user.id } });
    expect(await verifyPassword('my-first-real-password', after.passwordHash as string)).toBe(true);
    // Nothing was compromised and there are no password sessions to end, so
    // the version stays put and no replacement cookie is needed.
    expect(after.tokenVersion).toBe(user.tokenVersion);
    expect(setCookie(response)).toBeNull();
  });

  it('rejects a cross-origin post', async () => {
    const user = await seed('pw-csrf');
    const response = await changePassword(
      post('/api/auth/password', {
        cookie: await token(user.id, user.tokenVersion),
        origin: 'https://attacker.example',
        fields: {
          currentPassword: CURRENT,
          newPassword: 'a-brand-new-password-2026',
          confirmPassword: 'a-brand-new-password-2026',
        },
      }),
    );

    expect(response.status).toBe(403);
    const after = await prisma.user.findUniqueOrThrow({ where: { id: user.id } });
    expect(await verifyPassword(CURRENT, after.passwordHash as string)).toBe(true);
  });

  it('sends an unauthenticated or stale-cookie request to sign in', async () => {
    const user = await seed('pw-stale');
    const fields = {
      currentPassword: CURRENT,
      newPassword: 'a-brand-new-password-2026',
      confirmPassword: 'a-brand-new-password-2026',
    };

    const noCookie = await changePassword(post('/api/auth/password', { fields }));
    expect(noCookie.status).toBe(303);
    expect(noCookie.headers.get('location')).toContain('/sign-in');

    // A token whose version has moved on is not a session, however well signed.
    const stale = await changePassword(
      post('/api/auth/password', {
        cookie: await token(user.id, user.tokenVersion + 5),
        fields,
      }),
    );
    expect(stale.headers.get('location')).toContain('/sign-in');

    const after = await prisma.user.findUniqueOrThrow({ where: { id: user.id } });
    expect(await verifyPassword(CURRENT, after.passwordHash as string)).toBe(true);
  });
});

describe('POST /api/auth/sessions/revoke', () => {
  it('signs other devices out and re-issues this device a cookie', async () => {
    const user = await seed('revoke');
    const before = await token(user.id, user.tokenVersion);

    const response = await revokeSessions(post('/api/auth/sessions/revoke', { cookie: before }));

    expect(response.status).toBe(303);
    expect(outcome(response, 'sessions')).toBe('revoked');

    const after = await prisma.user.findUniqueOrThrow({ where: { id: user.id } });
    expect(after.tokenVersion).toBe(user.tokenVersion + 1);

    const reissued = setCookie(response);
    expect(reissued).toBeTruthy();
    expect((await readToken(reissued as string)).tokenVersion).toBe(after.tokenVersion);
    // The cookie the request came in with is exactly what was revoked.
    expect((await readToken(before)).tokenVersion).toBe(user.tokenVersion);
  });

  it('rejects a cross-origin post', async () => {
    const user = await seed('revoke-csrf');
    const response = await revokeSessions(
      post('/api/auth/sessions/revoke', {
        cookie: await token(user.id, user.tokenVersion),
        origin: 'https://attacker.example',
      }),
    );

    expect(response.status).toBe(403);
    const after = await prisma.user.findUniqueOrThrow({ where: { id: user.id } });
    expect(after.tokenVersion).toBe(user.tokenVersion);
  });
});
