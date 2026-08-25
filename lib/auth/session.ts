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

export async function setSessionCookie(token: string): Promise<void> {
  const store = await cookies();
  store.set(SESSION_COOKIE, token, {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'lax',
    path: '/',
    maxAge: SESSION_MAX_AGE_S,
  });
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
