/**
 * A stand-in Google, for the end-to-end suite.
 *
 * Real Google cannot be driven from a test: there is no way to log in as an
 * arbitrary person, and no way to make it assert an unverified email. So this
 * serves the three endpoints DiaLog actually talks to — authorize, token and
 * JWKS — and signs real RS256 ID tokens with a key it generates at startup.
 *
 * Nothing here is a stub of DiaLog's own code: the app still runs its full
 * PKCE, state, nonce, signature, issuer and audience checks, and this server
 * exists purely to be the other end of them. `POST /_identity` chooses which
 * person the next sign-in will assert.
 *
 * Loopback only, started by playwright.config.ts. Never runs in production —
 * see the hostname restriction in lib/auth/oauth/google.ts.
 */
import { createServer } from 'node:http';
import { createHash, randomUUID } from 'node:crypto';
import { SignJWT, exportJWK, generateKeyPair, type JWK } from 'jose';

const PORT = Number(process.env.FAKE_GOOGLE_PORT ?? 3210);
const ISSUER = `http://127.0.0.1:${PORT}`;
const KEY_ID = 'fake-google-key';

interface Identity {
  sub: string;
  email: string;
  emailVerified: boolean;
  name: string | null;
}

/** Whoever the next authorize call will sign in as. */
let identity: Identity = {
  sub: 'google-sub-default',
  email: 'default@example.com',
  emailVerified: true,
  name: 'Default Person',
};

interface PendingCode {
  challenge: string;
  nonce: string;
  identity: Identity;
}
const codes = new Map<string, PendingCode>();

function base64url(input: Buffer): string {
  return input.toString('base64url');
}

async function main(): Promise<void> {
  const { publicKey, privateKey } = await generateKeyPair('RS256', { extractable: true });
  const jwk: JWK = { ...(await exportJWK(publicKey)), kid: KEY_ID, alg: 'RS256', use: 'sig' };

  const server = createServer(async (request, response) => {
    const url = new URL(request.url ?? '/', ISSUER);

    // -- choose the person the next sign-in asserts
    if (request.method === 'POST' && url.pathname === '/_identity') {
      const body = await readBody(request);
      const next = JSON.parse(body || '{}') as Partial<Identity>;
      identity = {
        sub: next.sub ?? identity.sub,
        email: next.email ?? identity.email,
        emailVerified: next.emailVerified ?? true,
        name: next.name === undefined ? identity.name : next.name,
      };
      return json(response, 200, { ok: true, identity });
    }

    // -- the consent screen, minus the consent
    if (url.pathname === '/authorize') {
      const redirectUri = url.searchParams.get('redirect_uri');
      const state = url.searchParams.get('state') ?? '';
      const challenge = url.searchParams.get('code_challenge') ?? '';
      const nonce = url.searchParams.get('nonce') ?? '';
      if (!redirectUri) return json(response, 400, { error: 'missing redirect_uri' });

      // `?deny=1` on the app's start URL is not a thing; tests exercise the
      // cancel path by pointing the browser straight at the callback instead.
      const code = randomUUID();
      codes.set(code, { challenge, nonce, identity });

      const back = new URL(redirectUri);
      back.searchParams.set('code', code);
      back.searchParams.set('state', state);
      response.writeHead(302, { location: back.toString() });
      return response.end();
    }

    // -- code + verifier for an ID token
    if (request.method === 'POST' && url.pathname === '/token') {
      const params = new URLSearchParams(await readBody(request));
      const pending = codes.get(params.get('code') ?? '');
      if (!pending) return json(response, 400, { error: 'invalid_grant' });
      codes.delete(params.get('code') ?? ''); // single use, as Google does

      const verifier = params.get('code_verifier') ?? '';
      const computed = base64url(createHash('sha256').update(verifier).digest());
      if (computed !== pending.challenge) return json(response, 400, { error: 'invalid_grant' });
      if (!params.get('client_id') || !params.get('client_secret')) {
        return json(response, 401, { error: 'invalid_client' });
      }

      const idToken = await new SignJWT({
        email: pending.identity.email,
        email_verified: pending.identity.emailVerified,
        name: pending.identity.name ?? undefined,
        nonce: pending.nonce,
      })
        .setProtectedHeader({ alg: 'RS256', kid: KEY_ID })
        .setIssuer(ISSUER)
        .setAudience(params.get('client_id') as string)
        .setSubject(pending.identity.sub)
        .setIssuedAt()
        .setExpirationTime('5m')
        .sign(privateKey);

      return json(response, 200, { id_token: idToken, token_type: 'Bearer', expires_in: 300 });
    }

    if (url.pathname === '/certs') return json(response, 200, { keys: [jwk] });

    return json(response, 404, { error: 'not_found' });
  });

  server.listen(PORT, '127.0.0.1', () => {
    // Playwright's webServer waits for this port to accept connections.
    console.log(`fake google listening on ${ISSUER}`);
  });
}

function readBody(request: import('node:http').IncomingMessage): Promise<string> {
  return new Promise((resolve) => {
    let data = '';
    request.on('data', (chunk) => (data += chunk));
    request.on('end', () => resolve(data));
  });
}

function json(response: import('node:http').ServerResponse, status: number, body: unknown): void {
  response.writeHead(status, { 'content-type': 'application/json' });
  response.end(JSON.stringify(body));
}

void main();
