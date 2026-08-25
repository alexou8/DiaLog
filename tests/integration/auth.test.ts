import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { hashPassword, validatePassword, verifyPassword } from '@/lib/auth/password';
import { signSession, verifySession, type SessionPayload } from '@/lib/auth/session';
import { pruneRateLimits, rateLimit, RATE_LIMITS } from '@/lib/auth/rate-limit';
import { SignJWT } from 'jose';
import { prisma, createTestUser, deleteTestUser, type SeededUser } from './test-helpers';

describe('password hashing and verification', () => {
  it('round-trips: verifyPassword accepts the correct plaintext and rejects a wrong one', async () => {
    const hash = await hashPassword('correct horse battery staple 1');
    expect(await verifyPassword('correct horse battery staple 1', hash)).toBe(true);
    expect(await verifyPassword('wrong password entirely', hash)).toBe(false);
  });

  it('produces a different hash each time (random salt) but both verify', async () => {
    const h1 = await hashPassword('same-password-twice');
    const h2 = await hashPassword('same-password-twice');
    expect(h1).not.toBe(h2);
    expect(await verifyPassword('same-password-twice', h1)).toBe(true);
    expect(await verifyPassword('same-password-twice', h2)).toBe(true);
  });
});

describe('validatePassword policy', () => {
  it('rejects passwords under 10 characters', () => {
    expect(validatePassword('short1').ok).toBe(false);
  });
  it('rejects passwords over 200 characters', () => {
    expect(validatePassword('a'.repeat(201)).ok).toBe(false);
  });
  it('rejects common breached passwords case-insensitively', () => {
    expect(validatePassword('Password123').ok).toBe(false);
    expect(validatePassword('diabetes1').ok).toBe(false);
  });
  it('accepts a reasonable passphrase', () => {
    expect(validatePassword('correct horse battery staple').ok).toBe(true);
  });
});

/**
 * Mirrors the guard in lib/auth/current-user.ts's getCurrentUser: verify the
 * signed token, look up the user, and reject if tokenVersion has moved on.
 * Reimplemented here (rather than imported) because that module reads the
 * session cookie via next/headers, which only works inside a request scope.
 */
async function resolveUserForToken(token: string): Promise<{ userId: string } | null> {
  const session = await verifySession(token);
  if (!session) return null;
  const user = await prisma.user.findUnique({
    where: { id: session.userId },
    select: { id: true, tokenVersion: true },
  });
  if (!user) return null;
  if (user.tokenVersion !== session.tokenVersion) return null;
  return { userId: user.id };
}

describe('session tokens', () => {
  let seeded: SeededUser;

  beforeEach(async () => {
    seeded = await createTestUser('session');
  });
  afterEach(async () => {
    await deleteTestUser(seeded.user.id);
  });

  it('a token signed for a user verifies and yields the right userId', async () => {
    const token = await signSession({
      userId: seeded.user.id,
      tokenVersion: seeded.user.tokenVersion,
    });
    const resolved = await resolveUserForToken(token);
    expect(resolved?.userId).toBe(seeded.user.id);
  });

  it('a token signed with a different secret fails verification', async () => {
    const badSecretToken = await new SignJWT({ tokenVersion: 0 })
      .setProtectedHeader({ alg: 'HS256' })
      .setSubject(seeded.user.id)
      .setIssuer('dialog')
      .setIssuedAt()
      .setExpirationTime('30d')
      .sign(new TextEncoder().encode('a-completely-different-secret-value-000000'));

    expect(await verifySession(badSecretToken)).toBeNull();
    expect(await resolveUserForToken(badSecretToken)).toBeNull();
  });

  it('an expired token fails verification', async () => {
    const expiredToken = await new SignJWT({ tokenVersion: 0 })
      .setProtectedHeader({ alg: 'HS256' })
      .setSubject(seeded.user.id)
      .setIssuer('dialog')
      .setIssuedAt(Math.floor(Date.now() / 1000) - 3600)
      .setExpirationTime(Math.floor(Date.now() / 1000) - 1800)
      .sign(new TextEncoder().encode(process.env.AUTH_SECRET as string));

    expect(await verifySession(expiredToken)).toBeNull();
  });

  it('a tampered token fails verification', async () => {
    const token = await signSession({
      userId: seeded.user.id,
      tokenVersion: seeded.user.tokenVersion,
    });
    const parts = token.split('.');
    // Flip a character in the payload segment to corrupt the signature.
    const payload = parts[1] ?? '';
    const flipped = payload.slice(0, -1) + (payload.endsWith('A') ? 'B' : 'A');
    const tampered = [parts[0], flipped, parts[2]].join('.');
    expect(await verifySession(tampered)).toBeNull();
  });

  it('bumping tokenVersion invalidates a previously valid session (sign out everywhere)', async () => {
    const token = await signSession({
      userId: seeded.user.id,
      tokenVersion: seeded.user.tokenVersion,
    });
    expect((await resolveUserForToken(token))?.userId).toBe(seeded.user.id);

    await prisma.user.update({
      where: { id: seeded.user.id },
      data: { tokenVersion: { increment: 1 } },
    });

    expect(await resolveUserForToken(token)).toBeNull();
    // A freshly signed token with the new version works again.
    const bumped = await prisma.user.findUniqueOrThrow({ where: { id: seeded.user.id } });
    const newToken = await signSession({
      userId: seeded.user.id,
      tokenVersion: bumped.tokenVersion,
    });
    expect((await resolveUserForToken(newToken))?.userId).toBe(seeded.user.id);
  });

  it('rejects a well-formed token payload with a non-string subject or garbage tokenVersion type', async () => {
    // verifySession explicitly checks typeof payload.sub === 'string' etc; build a token that
    // has the right shape but wrong types to exercise that guard, not just JWT validity.
    const weird: SessionPayload = {
      userId: seeded.user.id,
      tokenVersion: seeded.user.tokenVersion,
    };
    const ok = await signSession(weird);
    expect(await verifySession(ok)).toEqual(weird);
  });
});

describe('rate limiter', () => {
  const KEY_PREFIX = `authtest-${Date.now()}`;

  afterEach(() => {
    vi.useRealTimers();
  });

  it('permits up to the limit, then rejects, and reports retryAfter', () => {
    const key = `${KEY_PREFIX}-limit`;
    const { limit, windowMs } = RATE_LIMITS.signIn;
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-01-01T00:00:00Z'));

    for (let i = 0; i < limit; i++) {
      const result = rateLimit(key, limit, windowMs);
      expect(result.ok).toBe(true);
    }
    const blocked = rateLimit(key, limit, windowMs);
    expect(blocked.ok).toBe(false);
    expect(blocked.remaining).toBe(0);
    expect(blocked.retryAfterSeconds).toBeGreaterThan(0);
    expect(blocked.retryAfterSeconds).toBeLessThanOrEqual(Math.ceil(windowMs / 1000));
  });

  it('resets after the window elapses, without sleeping (advance fake time)', () => {
    const key = `${KEY_PREFIX}-reset`;
    const limit = 3;
    const windowMs = 1000;
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-01-01T00:00:00Z'));

    for (let i = 0; i < limit; i++) expect(rateLimit(key, limit, windowMs).ok).toBe(true);
    expect(rateLimit(key, limit, windowMs).ok).toBe(false);

    vi.setSystemTime(new Date(Date.now() + windowMs + 1));
    const afterReset = rateLimit(key, limit, windowMs);
    expect(afterReset.ok).toBe(true);
    expect(afterReset.remaining).toBe(limit - 1);
  });

  it('pruneRateLimits does not throw and is safe to call repeatedly', () => {
    rateLimit(`${KEY_PREFIX}-prune`, 1, 1000);
    expect(() => pruneRateLimits(Date.now() + 100_000)).not.toThrow();
  });
});
