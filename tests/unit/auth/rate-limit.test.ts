/**
 * Per-account sign-in limiting.
 *
 * The per-IP budget alone let a distributed attacker spread guesses at one
 * account across many addresses. These pin the account key (stable, secret,
 * no raw email) and that the sign-in action enforces it independently of IP —
 * the real action, with Next's request APIs and the database mocked out.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const ip = vi.hoisted(() => ({ value: '198.51.100.1' }));
const db = vi.hoisted(() => ({ user: { findUnique: vi.fn(), update: vi.fn() } }));
const audit = vi.hoisted(() => vi.fn());
vi.mock('next/headers', () => ({
  headers: async () => new Headers({ 'x-forwarded-for': ip.value }),
  cookies: async () => ({ get: () => undefined, set: () => undefined }),
}));
vi.mock('@/lib/db/prisma', () => ({ prisma: db }));
vi.mock('@/lib/auth/audit', () => ({ audit }));
// A real bcrypt comparison per attempt would make this file take ~10 s while
// proving nothing about rate limiting; password-policy.test covers the hash.
vi.mock('@/lib/auth/password', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/lib/auth/password')>()),
  verifyPassword: vi.fn(async () => false),
  dummyPasswordHash: vi.fn(async () => '$2b$12$' + 'a'.repeat(53)),
}));

// A successful sign-in ends in redirect(); stand in a recognisable throw for
// Next's internal one so the test can tell success from a returned error.
vi.mock('next/navigation', () => ({
  redirect: (to: string) => {
    throw new Error(`REDIRECT:${to}`);
  },
}));

import { signInAction } from '@/lib/actions/auth';
import { verifyPassword } from '@/lib/auth/password';
import { RATE_LIMITS, accountRateLimitKey, pruneRateLimits } from '@/lib/auth/rate-limit';

const SECRET = 'unit-rate-limit-secret-at-least-32-characters';

beforeEach(() => {
  vi.stubEnv('AUTH_SECRET', SECRET);
  pruneRateLimits(Infinity);
  db.user.findUnique.mockReset();
  db.user.findUnique.mockResolvedValue(null);
});
afterEach(() => {
  vi.unstubAllEnvs();
  pruneRateLimits(Infinity);
  vi.mocked(verifyPassword).mockReset();
  vi.mocked(verifyPassword).mockResolvedValue(false);
});

function form(email: string): FormData {
  const data = new FormData();
  data.set('email', email);
  data.set('password', 'a-wrong-password-guess');
  return data;
}

let ipCounter = 0;
function freshIp(): void {
  ipCounter += 1;
  ip.value = `203.0.113.${ipCounter}`;
}

describe('accountRateLimitKey', () => {
  it('is stable across case and whitespace, and separated by scope', () => {
    const key = accountRateLimitKey('signin-account', 'Person@Example.com');
    expect(accountRateLimitKey('signin-account', '  person@example.com ')).toBe(key);
    const other = accountRateLimitKey('other-scope', 'person@example.com');
    expect(other.split(':')[1]).not.toBe(key.split(':')[1]);
  });

  it('never contains the address, and depends on the secret', () => {
    const key = accountRateLimitKey('signin-account', 'person@example.com');
    expect(key).not.toContain('person');
    expect(key).not.toContain('example.com');
    vi.stubEnv('AUTH_SECRET', `${SECRET}-rotated`);
    expect(accountRateLimitKey('signin-account', 'person@example.com')).not.toBe(key);
  });

  it('fails closed without a usable secret', () => {
    vi.stubEnv('AUTH_SECRET', 'short');
    expect(() => accountRateLimitKey('signin-account', 'person@example.com')).toThrow();
  });
});

describe('signInAction per-account limit', () => {
  it('stops guesses at one account spread across many IPs', async () => {
    const { limit } = RATE_LIMITS.signInAccount;
    for (let i = 0; i < limit; i++) {
      freshIp();
      const result = await signInAction(null, form('target@example.com'));
      expect(result.message).toMatch(/do not match an account/);
    }
    freshIp();
    const blocked = await signInAction(null, form('TARGET@example.com'));
    expect(blocked.ok).toBe(false);
    expect(blocked.message).toMatch(/Too many sign-in attempts/);
    // Refused before the account is even looked up.
    expect(db.user.findUnique).toHaveBeenCalledTimes(limit);
  });

  it('does not spill over to other accounts', async () => {
    for (let i = 0; i < RATE_LIMITS.signInAccount.limit + 1; i++) {
      freshIp();
      await signInAction(null, form('target@example.com'));
    }
    freshIp();
    const other = await signInAction(null, form('someone-else@example.com'));
    expect(other.message).toMatch(/do not match an account/);
  });

  it('does not count successful sign-ins, so a busy shared account is never locked', async () => {
    // The shared demo account is signed in to by every visitor; charging
    // successes locked it for everyone after 20 sign-ins in 15 minutes.
    db.user.findUnique.mockResolvedValue({
      id: 'user-1',
      tokenVersion: 0,
      passwordHash: '$2b$12$' + 'b'.repeat(53),
      profile: { onboardingCompletedAt: new Date() },
      identities: [],
    });
    db.user.update.mockResolvedValue({});
    vi.mocked(verifyPassword).mockResolvedValue(true);
    for (let i = 0; i < RATE_LIMITS.signInAccount.limit + 5; i++) {
      freshIp();
      await expect(signInAction(null, form('busy@example.com'))).rejects.toThrow('REDIRECT:/app');
    }
  });

  it('still locks after the failure budget is spent, even with a success earlier', async () => {
    db.user.findUnique.mockResolvedValue({
      id: 'user-1',
      tokenVersion: 0,
      passwordHash: '$2b$12$' + 'b'.repeat(53),
      profile: { onboardingCompletedAt: new Date() },
      identities: [],
    });
    db.user.update.mockResolvedValue({});
    vi.mocked(verifyPassword).mockResolvedValueOnce(true);
    freshIp();
    await expect(signInAction(null, form('busy@example.com'))).rejects.toThrow('REDIRECT:/app');
    for (let i = 0; i < RATE_LIMITS.signInAccount.limit; i++) {
      freshIp();
      const result = await signInAction(null, form('busy@example.com'));
      expect(result.message).toMatch(/do not match an account/);
    }
    freshIp();
    const blocked = await signInAction(null, form('busy@example.com'));
    expect(blocked.message).toMatch(/Too many sign-in attempts/);
  });

  it('keeps the per-IP limit for one client trying many accounts', async () => {
    ip.value = '192.0.2.77';
    for (let i = 0; i < RATE_LIMITS.signIn.limit; i++) {
      await signInAction(null, form(`person${i}@example.com`));
    }
    const blocked = await signInAction(null, form('yet-another@example.com'));
    expect(blocked.message).toMatch(/Too many sign-in attempts/);
  });
});
