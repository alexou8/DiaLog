/**
 * Session handling.
 *
 * DiaLog uses a stateless, signed session cookie (JWT via `jose`) rather than a
 * third-party auth service, so that health accounts are not dependent on an
 * external identity provider and no user data leaves the deployment. The token
 * carries only the user id and a `tokenVersion`; bumping the version in the
 * database invalidates every outstanding cookie ("sign out everywhere").
 */
import { SignJWT, jwtVerify } from 'jose';
import { cookies } from 'next/headers';

export const SESSION_COOKIE = 'dialog_session';
const SESSION_MAX_AGE_S = 60 * 60 * 24 * 30; // 30 days
const ISSUER = 'dialog';

export interface SessionPayload {
  userId: string;
  tokenVersion: number;
}

/**
 * Whether this deployment can mint sessions at all.
 *
 * `secret()` throwing is the right behaviour — a deployment that cannot sign a
 * cookie must not pretend to sign anyone in — but the throw has to happen
 * somewhere that can explain itself. Callers that run inside a server action or
 * a page render check this first and surface a configuration message, rather
 * than letting the raw error escape and replace the whole page with the
 * framework's generic "Application error" screen. See `lib/actions/auth.ts`.
 */
export function isSessionSecretConfigured(): boolean {
  const value = process.env.AUTH_SECRET;
  return typeof value === 'string' && value.length >= 32;
}

function secret(): Uint8Array {
  const value = process.env.AUTH_SECRET;
  if (!value || value.length < 32) {
    throw new Error('AUTH_SECRET is missing or too short (need at least 32 characters).');
  }
  return new TextEncoder().encode(value);
}

export async function signSession(payload: SessionPayload): Promise<string> {
  return new SignJWT({ tokenVersion: payload.tokenVersion })
    .setProtectedHeader({ alg: 'HS256' })
    .setSubject(payload.userId)
    .setIssuer(ISSUER)
    .setIssuedAt()
    .setExpirationTime(`${SESSION_MAX_AGE_S}s`)
    .sign(secret());
}

export async function verifySession(token: string): Promise<SessionPayload | null> {
  try {
    const { payload } = await jwtVerify(token, secret(), { issuer: ISSUER });
    if (typeof payload.sub !== 'string' || typeof payload.tokenVersion !== 'number') return null;
    return { userId: payload.sub, tokenVersion: payload.tokenVersion };
  } catch {
    return null;
  }
}

/**
 * Cookie attributes for the session cookie, in one place so that the two ways
 * of issuing it cannot drift apart: `setSessionCookie()` writes to the ambient
 * request store, while a route handler that answers with a redirect attaches
 * the cookie to that response directly (see lib/auth/route-form.ts).
 */
export const SESSION_COOKIE_OPTIONS = {
  httpOnly: true,
  secure: process.env.NODE_ENV === 'production',
  sameSite: 'lax',
  path: '/',
  maxAge: SESSION_MAX_AGE_S,
} as const;

export async function setSessionCookie(token: string): Promise<void> {
  const store = await cookies();
  store.set(SESSION_COOKIE, token, SESSION_COOKIE_OPTIONS);
}

export async function clearSessionCookie(): Promise<void> {
  const store = await cookies();
  store.set(SESSION_COOKIE, '', { httpOnly: true, path: '/', maxAge: 0 });
}

export async function readSessionCookie(): Promise<SessionPayload | null> {
  const store = await cookies();
  const token = store.get(SESSION_COOKIE)?.value;
  if (!token) return null;
  return verifySession(token);
}
