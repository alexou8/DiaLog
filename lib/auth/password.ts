import bcrypt from 'bcryptjs';

/** Cost factor. 12 is a reasonable 2020s default for interactive logins. */
const ROUNDS = 12;

export async function hashPassword(plain: string): Promise<string> {
  return bcrypt.hash(plain, ROUNDS);
}

export async function verifyPassword(plain: string, hash: string): Promise<boolean> {
  return bcrypt.compare(plain, hash);
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
  too_long: 'That password is too long. Please use 200 characters or fewer.',
  too_common: 'That password is too common. Please choose something less guessable.',
};

export function validatePassword(
  plain: string,
): { ok: true } | { ok: false; code: PasswordPolicyCode; message: string } {
  const fail = (code: PasswordPolicyCode) =>
    ({ ok: false, code, message: PASSWORD_POLICY_MESSAGES[code] }) as const;

  if (plain.length < 10) return fail('too_short');
  if (plain.length > 200) return fail('too_long');
  if (COMMON.has(plain.toLowerCase())) return fail('too_common');
  return { ok: true };
}
