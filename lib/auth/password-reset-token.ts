import { createHash, randomBytes } from 'node:crypto';
import { SignJWT, jwtVerify } from 'jose';
import { passwordResetTokenSchema } from '@/lib/validation';
import { isSessionSecretConfigured } from './session';

// Thirty minutes bounds mailbox/link exposure while allowing ordinary mail delays.
export const PASSWORD_RESET_TTL_MS = 30 * 60_000;
export function hashResetToken(raw: string): string {
  return createHash('sha256').update(raw).digest('hex');
}
export function createResetToken(now = new Date()) {
  const raw = randomBytes(32).toString('base64url');
  return {
    raw,
    tokenHash: hashResetToken(raw),
    expiresAt: new Date(now.getTime() + PASSWORD_RESET_TTL_MS),
  };
}
export function resetTokenHash(raw: unknown): string | null {
  const parsed = passwordResetTokenSchema.safeParse(raw);
  return parsed.success ? hashResetToken(parsed.data) : null;
}
export interface ResetProof {
  userId: string;
  tokenHash: string;
}

function key() {
  if (!isSessionSecretConfigured()) throw new Error('Account recovery is not configured.');
  return new TextEncoder().encode(process.env.AUTH_SECRET);
}

/**
 * The emailed raw token never returns in HTML, RSC props, or an error redirect.
 * A purpose-bound signed proof lets the native form retry without echoing it.
 * A leaked database hash alone cannot mint this proof; AUTH_SECRET is required.
 */
export async function signResetProof(proof: ResetProof, expiresAt: Date): Promise<string> {
  return new SignJWT({ tokenHash: proof.tokenHash })
    .setProtectedHeader({ alg: 'HS256' })
    .setSubject(proof.userId)
    .setIssuer('dialog')
    .setAudience('password-reset')
    .setIssuedAt()
    .setExpirationTime(Math.floor(expiresAt.getTime() / 1000))
    .sign(key());
}
export async function verifyResetProof(value: string): Promise<ResetProof | null> {
  if (value.length > 2048) return null;
  try {
    const { payload } = await jwtVerify(value, key(), {
      algorithms: ['HS256'],
      issuer: 'dialog',
      audience: 'password-reset',
    });
    return typeof payload.sub === 'string' &&
      typeof payload.tokenHash === 'string' &&
      /^[a-f0-9]{64}$/.test(payload.tokenHash)
      ? { userId: payload.sub, tokenHash: payload.tokenHash }
      : null;
  } catch {
    return null;
  }
}
