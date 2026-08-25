/**
 * Shared helpers for the integration/security suite. Import `prisma` from
 * here (never construct a second PrismaClient) so every test file talks to
 * the same connection pool, pointed at `dialog_test` by setup-env.ts.
 */
import { randomUUID } from 'node:crypto';
import type { GlucoseContext, Profile, User } from '@prisma/client';
import { prisma } from '@/lib/db/prisma';
import { hashPassword } from '@/lib/auth/password';

export { prisma };

/** Copies a Node Buffer's bytes into a plain, non-shared ArrayBuffer for `prepareImport`. */
export function toArrayBuffer(buf: Buffer): ArrayBuffer {
  const out = new ArrayBuffer(buf.byteLength);
  new Uint8Array(out).set(buf);
  return out;
}

/** Unique, obviously-fake email per test run so tests never collide. */
export function testEmail(label: string): string {
  return `test+${label}-${randomUUID()}@dialog.test`;
}

export interface SeededUser {
  user: User;
  profile: Profile;
}

/** Create a fully-onboarded user (with profile) for tests. */
export async function createTestUser(
  label: string,
  overrides: { password?: string; targetLowMgdl?: number; targetHighMgdl?: number } = {},
): Promise<SeededUser> {
  const passwordHash = await hashPassword(
    overrides.password ?? 'a-reasonably-long-test-password-1',
  );
  const user = await prisma.user.create({
    data: {
      email: testEmail(label),
      passwordHash,
      profile: {
        create: {
          displayName: label,
          targetLowMgdl: overrides.targetLowMgdl ?? 70,
          targetHighMgdl: overrides.targetHighMgdl ?? 180,
          onboardingCompletedAt: new Date(),
        },
      },
    },
    include: { profile: true },
  });
  // `include` above guarantees profile is non-null (create ensures it exists).
  const profile = user.profile as Profile;
  return { user, profile };
}

/** Delete a user and (via cascade) everything owned by them. Safe to call twice. */
export async function deleteTestUser(userId: string): Promise<void> {
  await prisma.user.deleteMany({ where: { id: userId } });
}

export async function createGlucoseReading(params: {
  userId: string;
  takenAt: Date;
  valueMgdl: number;
  context?: GlucoseContext;
  dedupeKey?: string;
}) {
  return prisma.glucoseReading.create({
    data: {
      userId: params.userId,
      takenAt: params.takenAt,
      valueMgdl: params.valueMgdl,
      context: params.context ?? 'RANDOM',
      dedupeKey: params.dedupeKey ?? randomUUID(),
    },
  });
}
