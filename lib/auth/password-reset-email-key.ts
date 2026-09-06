import { createHmac } from 'node:crypto';

/**
 * Rate-limit keys need a secret to resist address-wordlist matching against a
 * leaked bucket store. Domain separation keeps this use distinct from session
 * signing. This MUST NOT replace the bare SHA-256 persisted for reset tokens.
 * A full process compromise that also reveals AUTH_SECRET defeats this defence.
 */
export function keyedRecoveryEmailHash(email: string): string {
  const secret = process.env.AUTH_SECRET;
  if (!secret || secret.length < 32) throw new Error('Account recovery is not configured.');
  return createHmac('sha256', secret)
    .update('password-reset-email\0')
    .update(email.trim().toLowerCase())
    .digest('hex');
}
