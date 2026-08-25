/**
 * Google OAuth 2.0 / OpenID Connect client.
 *
 * Deliberately hand-rolled rather than pulled from an auth framework: DiaLog
 * already owns its session format (`lib/auth/session.ts`), and a federated
 * sign-in only needs to establish *which* user is at the keyboard before that
 * existing cookie is minted. Google is used to authenticate, never to store or
 * read health data, and no Google token is persisted after the callback.
 *
 * Flow: authorization code + PKCE (S256), with `state` and `nonce` held in
 * short-lived HttpOnly cookies and checked on the way back.
 */
import { createRemoteJWKSet, jwtVerify } from 'jose';

const AUTHORIZE_URL = 'https://accounts.google.com/o/oauth2/v2/auth';
const TOKEN_URL = 'https://oauth2.googleapis.com/token';
const JWKS_URL = 'https://www.googleapis.com/oauth2/v3/certs';
const ISSUERS = ['https://accounts.google.com', 'accounts.google.com'];

interface Endpoints {
  authorize: string;
  token: string;
  jwks: string;
  issuers: string[];
}

/**
 * Google's endpoints, unless `GOOGLE_OIDC_TEST_ISSUER` points at a stand-in
 * provider on loopback. The end-to-end suite runs the whole redirect round trip
 * — PKCE, state, nonce, real RS256 signature checks — against a fake issuer it
 * controls (tests/e2e/fake-google.ts), which is only possible if the endpoints
 * can move.
 *
 * The loopback restriction is the safety catch: the override cannot be pointed
 * at a host an attacker controls, so a leaked or injected value can only ever
 * degrade to "sign-in fails".
 */
function endpoints(): Endpoints {
  const override = process.env.GOOGLE_OIDC_TEST_ISSUER;
  if (override) {
    try {
      const url = new URL(override);
      if (url.hostname === '127.0.0.1' || url.hostname === 'localhost') {
        const base = override.replace(/\/$/, '');
        return {
          authorize: `${base}/authorize`,
          token: `${base}/token`,
          jwks: `${base}/certs`,
          issuers: [base],
        };
      }
    } catch {
      // Fall through to the real Google endpoints.
    }
  }
  return { authorize: AUTHORIZE_URL, token: TOKEN_URL, jwks: JWKS_URL, issuers: ISSUERS };
}

export interface GoogleConfig {
  clientId: string;
  clientSecret: string;
  redirectUri: string;
}

/**
 * Read configuration, or return null when Google sign-in is not set up. The
 * app is fully usable without it, so callers degrade instead of throwing.
 */
export function googleConfig(): GoogleConfig | null {
  const clientId = process.env.GOOGLE_CLIENT_ID;
  const clientSecret = process.env.GOOGLE_CLIENT_SECRET;
  if (!clientId || !clientSecret) return null;
  const base = process.env.NEXT_PUBLIC_APP_URL?.replace(/\/$/, '') ?? 'http://localhost:3000';
  return { clientId, clientSecret, redirectUri: `${base}/api/auth/google/callback` };
}

export function isGoogleEnabled(): boolean {
  return googleConfig() !== null;
}

// --------------------------------------------------------------------- PKCE

function base64url(bytes: Uint8Array): string {
  return Buffer.from(bytes).toString('base64url');
}

export function randomToken(bytes = 32): string {
  return base64url(crypto.getRandomValues(new Uint8Array(bytes)));
}

export async function codeChallenge(verifier: string): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(verifier));
  return base64url(new Uint8Array(digest));
}

// ----------------------------------------------------------------- requests

export function buildAuthorizeUrl(
  config: GoogleConfig,
  params: { state: string; nonce: string; challenge: string; loginHint?: string },
): string {
  const url = new URL(endpoints().authorize);
  url.search = new URLSearchParams({
    client_id: config.clientId,
    redirect_uri: config.redirectUri,
    response_type: 'code',
    // Identity only. DiaLog never asks for Gmail, Drive or contacts scopes.
    scope: 'openid email profile',
    state: params.state,
    nonce: params.nonce,
    code_challenge: params.challenge,
    code_challenge_method: 'S256',
    prompt: 'select_account',
    ...(params.loginHint ? { login_hint: params.loginHint } : {}),
  }).toString();
  return url.toString();
}

export async function exchangeCode(
  config: GoogleConfig,
  code: string,
  verifier: string,
): Promise<{ idToken: string } | null> {
  const response = await fetch(endpoints().token, {
    method: 'POST',
    headers: { 'content-type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({
      code,
      client_id: config.clientId,
      client_secret: config.clientSecret,
      redirect_uri: config.redirectUri,
      grant_type: 'authorization_code',
      code_verifier: verifier,
    }),
    cache: 'no-store',
  });
  if (!response.ok) return null;
  const body = (await response.json()) as { id_token?: string };
  return body.id_token ? { idToken: body.id_token } : null;
}

export interface GoogleIdentity {
  subject: string;
  email: string;
  emailVerified: boolean;
  name: string | null;
}

// One key set per endpoint, cached across requests by `jose` and refetched when
// the provider rotates keys.
const jwksCache = new Map<string, ReturnType<typeof createRemoteJWKSet>>();

function keySet(url: string): ReturnType<typeof createRemoteJWKSet> {
  let existing = jwksCache.get(url);
  if (!existing) {
    existing = createRemoteJWKSet(new URL(url));
    jwksCache.set(url, existing);
  }
  return existing;
}

/**
 * Verify the ID token's signature, issuer, audience and nonce. Everything the
 * callback trusts about the person comes from here — never from query
 * parameters, which the browser controls.
 */
export async function verifyIdToken(
  config: GoogleConfig,
  idToken: string,
  nonce: string,
): Promise<GoogleIdentity | null> {
  try {
    const { jwks: jwksUrl, issuers } = endpoints();
    const { payload } = await jwtVerify(idToken, keySet(jwksUrl), {
      issuer: issuers,
      audience: config.clientId,
    });
    if (payload.nonce !== nonce) return null;
    const subject = typeof payload.sub === 'string' ? payload.sub : null;
    const email = typeof payload.email === 'string' ? payload.email.toLowerCase() : null;
    if (!subject || !email) return null;
    return {
      subject,
      email,
      emailVerified: payload.email_verified === true,
      name: typeof payload.name === 'string' ? payload.name : null,
    };
  } catch {
    return null;
  }
}
