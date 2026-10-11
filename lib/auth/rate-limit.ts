/**
 * Fixed-window rate limiting.
 *
 * The in-memory store is per server instance. On a single long-lived server
 * that is a real limit; on serverless (Vercel) every warm instance has its own
 * Map and a cold start empties it, so each budget below is really "per
 * instance", and an attacker who fans requests out multiplies it by however
 * many instances are warm. It still blunts naive scripts, but it is not a hard
 * ceiling. The interface is deliberately narrow so it can be swapped for a
 * shared store (a Postgres table, or Redis / Vercel KV) without touching call
 * sites — see docs/SECURITY.md, "Rate limiting — and its real limitation".
 */
import { createHmac } from 'node:crypto';

interface Bucket {
  count: number;
  resetAt: number;
}

const buckets = new Map<string, Bucket>();

export interface RateLimitResult {
  ok: boolean;
  remaining: number;
  retryAfterSeconds: number;
}

export function rateLimit(key: string, limit: number, windowMs: number): RateLimitResult {
  const now = Date.now();
  const existing = buckets.get(key);
  if (!existing || existing.resetAt <= now) {
    buckets.set(key, { count: 1, resetAt: now + windowMs });
    return { ok: true, remaining: limit - 1, retryAfterSeconds: 0 };
  }
  existing.count += 1;
  if (existing.count > limit) {
    return {
      ok: false,
      remaining: 0,
      retryAfterSeconds: Math.ceil((existing.resetAt - now) / 1000),
    };
  }
  return { ok: true, remaining: limit - existing.count, retryAfterSeconds: 0 };
}

/**
 * Forget a bucket. Used where a budget should only count failures: the caller
 * still charges every attempt up front (so concurrent requests cannot all slip
 * past the check before any is counted) and refunds the whole bucket once an
 * attempt succeeds.
 */
export function resetRateLimit(key: string): void {
  buckets.delete(key);
}

/** Opportunistic cleanup so the map cannot grow without bound. */
export function pruneRateLimits(now = Date.now()): void {
  for (const [key, bucket] of buckets) if (bucket.resetAt <= now) buckets.delete(key);
}

/**
 * Bucket key for one account, derived from the email typed into a form.
 *
 * Keyed by the address rather than a user id so an unknown address is limited
 * exactly like a known one; anything else would make the limiter itself an
 * account-existence oracle. HMAC'd with AUTH_SECRET so the bucket map never
 * holds raw addresses, and a dump of it cannot be matched against a list of
 * addresses without the secret. `scope` is mixed into the MAC so the sign-in
 * and any future per-account buckets can never collide. Callers must already
 * have checked that AUTH_SECRET is configured (both credential actions do).
 */
export function accountRateLimitKey(scope: string, email: string): string {
  const secret = process.env.AUTH_SECRET;
  if (!secret || secret.length < 32) throw new Error('Rate limiting is not configured.');
  const digest = createHmac('sha256', secret)
    .update(`rate-limit-account\0${scope}\0`)
    .update(email.trim().toLowerCase())
    .digest('hex');
  return `${scope}:${digest}`;
}

export const RATE_LIMITS = {
  signIn: { limit: 10, windowMs: 15 * 60_000 },
  // Per ACCOUNT (by email), on top of the per-IP budget above. The IP budget
  // stops one client trying many accounts; this one stops many clients (a
  // botnet, or one client rotating addresses) trying one account. Looser than
  // the IP budget because anyone can spend it on a victim's behalf: at 20 per
  // 15 minutes a deliberate lockout lasts at most a quarter of an hour and
  // leaves password recovery and Google sign-in working, while a distributed
  // guesser gets ~80 guesses an hour (per instance — see the header) against a
  // password of at least 10 characters. A successful sign-in clears the
  // bucket, so only failures accumulate: otherwise ordinary traffic to one
  // account (the shared demo, whose recovery and Google paths are disabled)
  // locked it out for everyone.
  signInAccount: { limit: 20, windowMs: 15 * 60_000 },
  // Starting a redirect to Google is not a credential guess — there is nothing
  // to brute force — so this gets its own, looser budget rather than spending
  // the sign-in one. It exists only to stop the endpoint being used to mint
  // attempt cookies in bulk. Shared-NAT offices and the e2e suite both make
  // more than ten of these in a quarter of an hour.
  oauthStart: { limit: 30, windowMs: 15 * 60_000 },
  signUp: { limit: 5, windowMs: 60 * 60_000 },
  passwordResetIp: { limit: 10, windowMs: 15 * 60_000 },
  passwordResetEmail: { limit: 3, windowMs: 60 * 60_000 },
  passwordResetSubmit: { limit: 10, windowMs: 15 * 60_000 },
  import: { limit: 20, windowMs: 60 * 60_000 },
  ai: { limit: 30, windowMs: 60 * 60_000 },
  write: { limit: 240, windowMs: 60 * 60_000 },
} as const;
