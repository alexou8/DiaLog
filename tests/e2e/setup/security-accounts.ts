/**
 * Provisions the accounts used by account-security.spec.ts.
 *
 * These accounts are created directly in the database and handed a signed
 * session cookie, rather than being registered through the sign-up form, for
 * two reasons.
 *
 * The first is budget. Sign-up is rate limited to 5 attempts per hour per
 * client (lib/auth/rate-limit.ts) and the suite already spends all five —
 * three in auth.setup.ts and two in auth-and-onboarding.spec.ts. A spec that
 * registered its own accounts would not just fail; it would take those two
 * unrelated tests down with it.
 *
 * The second is isolation. Every test in that spec revokes its own session on
 * purpose — that is the behaviour under test — so a shared `storageState`
 * would be dead for whichever test ran next. One disposable account each keeps
 * them independent of order.
 *
 * Minting the cookie here mirrors lib/auth/session.ts. That duplication is
 * deliberate and narrow: it keeps the setup free of Next's request-scoped
 * `cookies()` API, and tests/integration/auth.test.ts already signs tokens the
 * same way.
 */
import { readFileSync, mkdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { PrismaClient } from '@prisma/client';
import bcrypt from 'bcryptjs';
import { SignJWT } from 'jose';
import { AUTH_DIR } from './auth-state';
import { TEST_PASSWORD } from '../helpers';

const SESSION_COOKIE = 'dialog_session';
const SESSION_MAX_AGE_S = 60 * 60 * 24 * 30;

/**
 * The app server reads .env itself; this process does not, so read the one
 * value needed to sign a cookie the server will accept.
 */
function authSecret(): string {
  if (process.env.AUTH_SECRET) return process.env.AUTH_SECRET;
  const env = readFileSync(path.join(process.cwd(), '.env'), 'utf8');
  const secret = /^AUTH_SECRET\s*=\s*"?([^"\n\r]+)"?/m.exec(env)?.[1];
  if (!secret) throw new Error('AUTH_SECRET not found in the environment or in .env');
  return secret;
}

async function signSession(userId: string, tokenVersion: number): Promise<string> {
  return new SignJWT({ tokenVersion })
    .setProtectedHeader({ alg: 'HS256' })
    .setSubject(userId)
    .setIssuer('dialog')
    .setIssuedAt()
    .setExpirationTime(`${SESSION_MAX_AGE_S}s`)
    .sign(new TextEncoder().encode(authSecret()));
}

/**
 * Create one onboarded account with a known password and write a storageState
 * file holding a valid session cookie for it.
 */
export async function provisionSecurityAccount(
  prisma: PrismaClient,
  label: string,
  baseURL: string,
): Promise<{ email: string; statePath: string }> {
  const email = `e2e.security.${label}@dialog.test`;

  await prisma.user.deleteMany({ where: { email } });
  const user = await prisma.user.create({
    data: {
      email,
      passwordHash: await bcrypt.hash(TEST_PASSWORD, 10),
      profile: {
        create: {
          displayName: `Security ${label}`,
          // mg/dL to match the rest of the suite's fixtures.
          glucoseUnit: 'MGDL',
          onboardingCompletedAt: new Date(),
        },
      },
    },
  });

  const token = await signSession(user.id, user.tokenVersion);
  const { hostname } = new URL(baseURL);

  mkdirSync(AUTH_DIR, { recursive: true });
  const statePath = path.join(AUTH_DIR, `security-${label}.json`);
  writeFileSync(
    statePath,
    JSON.stringify({
      cookies: [
        {
          name: SESSION_COOKIE,
          value: token,
          domain: hostname,
          path: '/',
          expires: Math.floor(Date.now() / 1000) + SESSION_MAX_AGE_S,
          httpOnly: true,
          secure: false,
          sameSite: 'Lax',
        },
      ],
      origins: [],
    }),
    'utf8',
  );

  return { email, statePath };
}

/** The accounts account-security.spec.ts expects, one per test. */
export const SECURITY_LABELS = ['rounds', 'refuse', 'revoke-password', 'revoke-sessions'] as const;

export type SecurityLabel = (typeof SECURITY_LABELS)[number];

export function securityState(label: SecurityLabel): string {
  return path.join(AUTH_DIR, `security-${label}.json`);
}
