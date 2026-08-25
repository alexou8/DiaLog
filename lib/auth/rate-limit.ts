/**
 * Fixed-window rate limiting.
 *
 * The in-memory store is per server instance, which is enough to blunt
 * credential stuffing and import abuse on a single-region deployment. The
 * interface is deliberately narrow so it can be swapped for a shared store
 * (Redis / Vercel KV) without touching call sites — see docs/SECURITY.md.
 */
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

/** Opportunistic cleanup so the map cannot grow without bound. */
export function pruneRateLimits(now = Date.now()): void {
  for (const [key, bucket] of buckets) if (bucket.resetAt <= now) buckets.delete(key);
}

export const RATE_LIMITS = {
  signIn: { limit: 10, windowMs: 15 * 60_000 },
  signUp: { limit: 5, windowMs: 60 * 60_000 },
  import: { limit: 20, windowMs: 60 * 60_000 },
  ai: { limit: 30, windowMs: 60 * 60_000 },
  write: { limit: 240, windowMs: 60 * 60_000 },
} as const;
