/**
 * The short-lived cookie that carries a sign-in attempt from the redirect out
 * to Google and back again.
 *
 * Everything the callback needs — the CSRF `state`, the PKCE verifier, the
 * OIDC nonce, whether this was a sign-in or a link, and where to land
 * afterwards — travels in one signed, HttpOnly cookie rather than five plain
 * ones, so the callback can validate the whole attempt in a single check.
 */
import { SignJWT, jwtVerify } from 'jose';

export const OAUTH_COOKIE = 'dialog_oauth';
/** Long enough to pick an account and type a password, short enough to matter. */
export const OAUTH_MAX_AGE_S = 10 * 60;
const ISSUER = 'dialog-oauth';

export interface OAuthAttempt {
  state: string;
  verifier: string;
  nonce: string;
  mode: 'signin' | 'link';
  next: string;
}

function secret(): Uint8Array {
  const value = process.env.AUTH_SECRET;
  if (!value || value.length < 32) {
    throw new Error('AUTH_SECRET is missing or too short (need at least 32 characters).');
  }
  return new TextEncoder().encode(value);
}

export async function sealAttempt(attempt: OAuthAttempt): Promise<string> {
  return new SignJWT({ ...attempt })
    .setProtectedHeader({ alg: 'HS256' })
    .setIssuer(ISSUER)
    .setIssuedAt()
    .setExpirationTime(`${OAUTH_MAX_AGE_S}s`)
    .sign(secret());
}

export async function openAttempt(token: string | undefined): Promise<OAuthAttempt | null> {
  if (!token) return null;
  try {
    const { payload } = await jwtVerify(token, secret(), { issuer: ISSUER });
    const { state, verifier, nonce, mode, next } = payload as Record<string, unknown>;
    if (typeof state !== 'string' || typeof verifier !== 'string' || typeof nonce !== 'string') {
      return null;
    }
    if (mode !== 'signin' && mode !== 'link') return null;
    return { state, verifier, nonce, mode, next: typeof next === 'string' ? next : '/app' };
  } catch {
    return null;
  }
}

/**
 * Only same-origin paths may be used as a post-sign-in destination, so a
 * crafted link cannot bounce a freshly authenticated user off-site.
 */
export function safeNext(value: string | null | undefined, fallback: string): string {
  if (!value || !value.startsWith('/') || value.startsWith('//')) return fallback;
  return value;
}
