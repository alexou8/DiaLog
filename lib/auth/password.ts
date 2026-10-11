import { randomBytes } from 'node:crypto';
import bcrypt from 'bcryptjs';

/** Cost factor. 12 is a reasonable 2020s default for interactive logins. */
export const PASSWORD_HASH_ROUNDS = 12;

export async function hashPassword(plain: string): Promise<string> {
  return bcrypt.hash(plain, PASSWORD_HASH_ROUNDS);
}

export async function verifyPassword(plain: string, hash: string): Promise<boolean> {
  return bcrypt.compare(plain, hash);
}

let dummyHash: Promise<string> | null = null;

/**
 * A real bcrypt hash, at the real cost, of a random password nobody knows.
 *
 * Sign-in compares against this when the email matches no account, so that a
 * missing account costs one full bcrypt comparison exactly like a wrong
 * password does. It used to be a hand-typed string of the wrong length, and
 * bcryptjs returns `false` immediately for any hash that is not 60 characters
 * — so an unknown email answered in ~0 ms against ~250 ms for a known one,
 * which is an account-existence oracle. Generated once per process (one hash
 * at startup cost, paid lazily by the first unknown-email sign-in) rather than
 * pasted as a constant, so it can never drift from PASSWORD_HASH_ROUNDS.
 */
export function dummyPasswordHash(): Promise<string> {
  dummyHash ??= bcrypt.hash(randomBytes(32).toString('base64url'), PASSWORD_HASH_ROUNDS);
  return dummyHash;
}

/**
 * Password policy: length beats composition rules. We require a reasonable
 * minimum length and reject the handful of passwords that dominate breach
 * lists, rather than forcing symbol soup that users write on a sticky note.
 */
const COMMON = new Set([
  'password',
  'password1',
  'password123',
  '12345678',
  '123456789',
  'qwerty123',
  'letmein123',
  'welcome123',
  'iloveyou1',
  'admin1234',
  'diabetes1',
  'sunshine1',
]);

export type PasswordPolicyCode = 'too_short' | 'too_long' | 'too_common';

/**
 * Why a password was rejected, as a stable code plus the message shown to the
 * user. The code exists because the settings flow reports failures across a
 * redirect (see app/api/auth/password/route.ts) and a query string is no place
 * for a sentence: the handler sends the code, and the page turns it back into
 * this same message. Both halves therefore stay defined here.
 */
export const PASSWORD_POLICY_MESSAGES: Record<PasswordPolicyCode, string> = {
  too_short: 'Please use at least 10 characters. A short phrase works well.',
  too_long:
    'That password is too long. Please keep it to 72 characters or fewer, or fewer still if it uses accented letters, emoji or non-Latin script.',
  too_common: 'That password is too common. Please choose something less guessable.',
};

/**
 * bcrypt only ever reads the first 72 BYTES of its input and silently ignores
 * the rest, so a longer password is not stronger — two passwords sharing a
 * 72-byte prefix are the same password. Refusing it up front is more honest
 * than accepting characters that do nothing. Bytes, not characters: "é" is two
 * bytes in UTF-8 and most emoji are four, so 37 accented letters, or 19 emoji,
 * are already over.
 */
export const PASSWORD_MAX_BYTES = 72;

export function validatePassword(
  plain: string,
): { ok: true } | { ok: false; code: PasswordPolicyCode; message: string } {
  const fail = (code: PasswordPolicyCode) =>
    ({ ok: false, code, message: PASSWORD_POLICY_MESSAGES[code] }) as const;

  if (plain.length < 10) return fail('too_short');
  if (new TextEncoder().encode(plain).length > PASSWORD_MAX_BYTES) return fail('too_long');
  if (COMMON.has(plain.toLowerCase())) return fail('too_common');
  return { ok: true };
}
