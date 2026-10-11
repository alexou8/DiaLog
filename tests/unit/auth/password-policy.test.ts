/**
 * Two bcrypt facts the auth code depends on, pinned without a database.
 *
 *   - bcrypt reads only the first 72 BYTES of a password, so the policy caps
 *     there (in UTF-8 bytes, not characters) rather than accepting characters
 *     that silently do nothing.
 *   - Sign-in compares unknown emails against a dummy hash. bcryptjs returns
 *     `false` at once for anything that is not a well-formed 60-character
 *     hash, so a malformed dummy made unknown emails answer ~250 ms faster
 *     than known ones: an account-existence timing oracle.
 */
import bcrypt from 'bcryptjs';
import { describe, expect, it } from 'vitest';
import {
  PASSWORD_HASH_ROUNDS,
  PASSWORD_MAX_BYTES,
  PASSWORD_POLICY_MESSAGES,
  dummyPasswordHash,
  hashPassword,
  validatePassword,
} from '@/lib/auth/password';

describe('validatePassword: the 72-byte bcrypt ceiling', () => {
  it('accepts exactly 72 ASCII characters and refuses 73', () => {
    expect(validatePassword('a'.repeat(72)).ok).toBe(true);
    expect(validatePassword('a'.repeat(73))).toEqual({
      ok: false,
      code: 'too_long',
      message: PASSWORD_POLICY_MESSAGES.too_long,
    });
  });

  it('counts UTF-8 bytes, not characters', () => {
    // "é" is two bytes: 36 of them is 72 bytes, 37 is 74 bytes in 37 characters.
    expect(validatePassword('é'.repeat(36)).ok).toBe(true);
    expect(validatePassword('é'.repeat(37))).toMatchObject({ ok: false, code: 'too_long' });
    // Most emoji are four bytes: 19 of them is 76 bytes in well under 72 characters.
    const emoji = '\u{1F9C1}';
    expect(validatePassword(emoji.repeat(18)).ok).toBe(true);
    expect(validatePassword(emoji.repeat(19))).toMatchObject({ ok: false, code: 'too_long' });
  });

  it('still refuses what it refused before', () => {
    expect(validatePassword('short1')).toMatchObject({ ok: false, code: 'too_short' });
    expect(validatePassword('x'.repeat(201))).toMatchObject({ ok: false, code: 'too_long' });
    expect(validatePassword('Password123')).toMatchObject({ ok: false, code: 'too_common' });
  });

  it('says why in terms a person can act on', () => {
    expect(PASSWORD_MAX_BYTES).toBe(72);
    expect(PASSWORD_POLICY_MESSAGES.too_long).toMatch(/72 characters/);
  });
});

describe('dummyPasswordHash', () => {
  it('is a real 60-character bcrypt hash at the configured cost', async () => {
    const dummy = await dummyPasswordHash();
    expect(dummy).toHaveLength(60);
    expect(dummy).toMatch(/^\$2[aby]\$\d{2}\$[./A-Za-z0-9]{53}$/);
    expect(bcrypt.getRounds(dummy)).toBe(PASSWORD_HASH_ROUNDS);
    // The same cost a real account's hash has, which is what equalises timing.
    expect(bcrypt.getRounds(dummy)).toBe(bcrypt.getRounds(await hashPassword('any-password-1')));
  });

  it('is generated once per process and matches no guessable password', async () => {
    expect(await dummyPasswordHash()).toBe(await dummyPasswordHash());
    expect(await bcrypt.compare('', await dummyPasswordHash())).toBe(false);
    expect(await bcrypt.compare('password', await dummyPasswordHash())).toBe(false);
  });
});
