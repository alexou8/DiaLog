/**
 * The demo account is recognised by email (there is no isDemo column), so the
 * constant here and the one prisma/seed.ts creates the account with must not
 * drift apart — if they did, every demo lock would silently stop applying.
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { DEMO_EMAIL, DEMO_READ_ONLY_MESSAGE, isDemoEmail } from '@/lib/auth/demo';
import { oauthMessage } from '@/lib/auth/oauth/link';

describe('demo account identity', () => {
  it('matches the address prisma/seed.ts seeds', () => {
    const seed = readFileSync(path.resolve(__dirname, '../../../prisma/seed.ts'), 'utf8');
    const match = /const DEMO_EMAIL = '([^']+)'/.exec(seed);
    expect(match?.[1]).toBe(DEMO_EMAIL);
  });

  it('matches regardless of case and surrounding space, and nothing else', () => {
    expect(isDemoEmail(DEMO_EMAIL)).toBe(true);
    expect(isDemoEmail(`  ${DEMO_EMAIL.toUpperCase()} `)).toBe(true);
    expect(isDemoEmail('demo@dialog.health.evil.example')).toBe(false);
    expect(isDemoEmail('someone@dialog.health')).toBe(false);
    expect(isDemoEmail(null)).toBe(false);
    expect(isDemoEmail(undefined)).toBe(false);
    expect(isDemoEmail('')).toBe(false);
  });

  it('is explained on Settings through the ?error= channel', () => {
    expect(oauthMessage('demo_read_only')).toBe(DEMO_READ_ONLY_MESSAGE);
  });
});
