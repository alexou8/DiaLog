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

export function validatePassword(plain: string): { ok: true } | { ok: false; message: string } {
  if (plain.length < 10) {
    return { ok: false, message: 'Please use at least 10 characters. A short phrase works well.' };
  }
  if (plain.length > 200) {
    return { ok: false, message: 'That password is too long. Please use 200 characters or fewer.' };
  }
  if (COMMON.has(plain.toLowerCase())) {
    return {
      ok: false,
      message: 'That password is too common. Please choose something less guessable.',
    };
  }
  return { ok: true };
}
