import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { isSessionSecretConfigured, signSession } from '@/lib/auth/session';

/**
 * A deployment missing `AUTH_SECRET` used to surface as a blank "Application
 * error" page: `signSession()` threw inside the sign-in server action and the
 * throw escaped uncaught. Signing must still refuse, but callers need a way to
 * ask first so they can explain themselves.
 */
describe('isSessionSecretConfigured', () => {
  const original = process.env.AUTH_SECRET;

  beforeEach(() => {
    delete process.env.AUTH_SECRET;
  });

  afterEach(() => {
    if (original === undefined) delete process.env.AUTH_SECRET;
    else process.env.AUTH_SECRET = original;
  });

  it('is false when the secret is absent', () => {
    expect(isSessionSecretConfigured()).toBe(false);
  });

  it('is false when the secret is shorter than 32 characters', () => {
    process.env.AUTH_SECRET = 'a'.repeat(31);
    expect(isSessionSecretConfigured()).toBe(false);
  });

  it('is true at exactly the minimum length', () => {
    process.env.AUTH_SECRET = 'a'.repeat(32);
    expect(isSessionSecretConfigured()).toBe(true);
  });

  it('agrees with what signSession will accept', async () => {
    expect(isSessionSecretConfigured()).toBe(false);
    await expect(signSession({ userId: 'u1', tokenVersion: 0 })).rejects.toThrow(/AUTH_SECRET/);

    process.env.AUTH_SECRET = 'x'.repeat(48);
    expect(isSessionSecretConfigured()).toBe(true);
    await expect(signSession({ userId: 'u1', tokenVersion: 0 })).resolves.toEqual(
      expect.any(String),
    );
  });
});
