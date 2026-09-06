import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  createResetToken,
  hashResetToken,
  PASSWORD_RESET_TTL_MS,
  resetTokenHash,
  signResetProof,
  verifyResetProof,
} from '@/lib/auth/password-reset-token';
import { validatePassword } from '@/lib/auth/password';
import { keyedRecoveryEmailHash } from '@/lib/auth/password-reset-email-key';
import { forgotPasswordSchema, resetPasswordSchema } from '@/lib/validation';

afterEach(() => {
  vi.unstubAllEnvs();
  vi.useRealTimers();
});
beforeEach(() => {
  vi.stubEnv('AUTH_SECRET', 'unit-recovery-secret-at-least-32-characters');
});

describe('password reset credentials', () => {
  it('HMACs normalized email keys without changing persisted token hashing', () => {
    const first = keyedRecoveryEmailHash(' PERSON@EXAMPLE.COM ');
    expect(first).toBe(keyedRecoveryEmailHash('person@example.com'));
    expect(first).not.toBe(hashResetToken('person@example.com'));
    expect(first).toMatch(/^[a-f0-9]{64}$/);
    vi.stubEnv('AUTH_SECRET', 'a-different-recovery-secret-at-least-32-characters');
    expect(keyedRecoveryEmailHash('person@example.com')).not.toBe(first);
    expect(hashResetToken('abc')).toBe(
      'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad',
    );
    vi.stubEnv('AUTH_SECRET', '');
    expect(() => keyedRecoveryEmailHash('person@example.com')).toThrow(
      'Account recovery is not configured.',
    );
  });
  it('uses 32 random bytes, base64url, and stores only a SHA-256 digest', () => {
    const first = createResetToken();
    const second = createResetToken();
    expect(Buffer.from(first.raw, 'base64url')).toHaveLength(32);
    expect(first.raw).toMatch(/^[A-Za-z0-9_-]{43}$/);
    expect(first.raw).not.toBe(second.raw);
    expect(first.tokenHash).toBe(hashResetToken(first.raw));
    expect(first.tokenHash).toHaveLength(64);
    expect(first.tokenHash).not.toContain(first.raw);
    expect(hashResetToken('abc')).toBe(
      'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad',
    );
  });
  it('expires exactly thirty minutes after issuance', () => {
    const now = new Date('2026-09-06T23:50:00Z');
    expect(PASSWORD_RESET_TTL_MS).toBe(1_800_000);
    expect(createResetToken(now).expiresAt.toISOString()).toBe('2026-09-07T00:20:00.000Z');
  });
  it.each([undefined, null, '', 'a'.repeat(42), 'a'.repeat(44), '!'.repeat(43), ['a'.repeat(43)]])(
    'rejects malformed raw credentials',
    (value) => {
      expect(resetTokenHash(value)).toBeNull();
    },
  );
  it('binds a signed form proof to its owner, purpose and original expiry', async () => {
    vi.useFakeTimers();
    const token = createResetToken();
    const claims = { userId: 'owner-a', tokenHash: token.tokenHash };
    const proof = await signResetProof(claims, token.expiresAt);
    expect(proof).not.toContain(token.raw);
    expect(await verifyResetProof(proof)).toEqual(claims);
    const parts = proof.split('.');
    parts[1] = Buffer.from(JSON.stringify({ sub: 'owner-b', tokenHash: token.tokenHash })).toString(
      'base64url',
    );
    expect(await verifyResetProof(parts.join('.'))).toBeNull();
    vi.advanceTimersByTime(PASSWORD_RESET_TTL_MS);
    expect(await verifyResetProof(proof)).toBeNull();
  });
  it('fails closed without a configured signing secret', async () => {
    vi.stubEnv('AUTH_SECRET', 'short');
    expect(await verifyResetProof('anything')).toBeNull();
    await expect(
      signResetProof({ userId: 'owner', tokenHash: 'a'.repeat(64) }, new Date()),
    ).rejects.toThrow('Account recovery is not configured.');
  });
});

describe('recovery validation', () => {
  it('normalizes email before lookup or throttling', () => {
    expect(forgotPasswordSchema.parse({ email: '  PERSON@EXAMPLE.COM ' })).toEqual({
      email: 'person@example.com',
    });
    expect(forgotPasswordSchema.safeParse({ email: 'invalid' }).success).toBe(false);
    expect(resetPasswordSchema.safeParse({ newPassword: 'a-password' }).success).toBe(false);
  });
  it.each([
    ['short', 'too_short'],
    ['x'.repeat(201), 'too_long'],
    ['password123', 'too_common'],
  ])('retains the existing password policy', (password, code) => {
    expect(validatePassword(password)).toMatchObject({ ok: false, code });
  });
  it('accepts a sufficiently long uncommon phrase', () => {
    expect(validatePassword('bright lanterns cross quiet rivers')).toEqual({ ok: true });
  });
});
