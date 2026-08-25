import { PrismaClient } from '@prisma/client';

/**
 * Reset the database to a known state before the suite runs.
 *
 * The end-to-end suite drives a real application against a real database, so
 * every test's starting point is whatever the last run left behind. That made
 * the suite non-deterministic in a way no individual assertion could fix:
 * records accumulated across runs, and content-addressed dedupe keys (see
 * lib/domain/dedupe.ts) meant re-creating "the same" record was rejected as a
 * duplicate rather than saved.
 *
 * Deleting every user cascades to every health record, import batch, insight
 * and conversation, so this leaves an empty database. The demo account is then
 * re-seeded, because the specs that only read data expect it.
 *
 * This is one of three layers that keep the suite deterministic:
 *   1. this reset — runs never inherit the previous run's data;
 *   2. per-attempt unique content (tests/e2e/unique.ts) — retries within a run
 *      cannot collide with their own earlier attempt;
 *   3. serial execution (playwright.config.ts) — specs sharing an account
 *      cannot interleave their writes.
 */
export default async function globalSetup(): Promise<void> {
  const prisma = new PrismaClient();
  try {
    await prisma.user.deleteMany({});
    await prisma.auditEvent.deleteMany({});
  } finally {
    await prisma.$disconnect();
  }

  // Re-seed the demo account the read-only specs rely on.
  const { execFileSync } = await import('node:child_process');
  execFileSync('npx', ['tsx', 'prisma/seed.ts'], { stdio: 'inherit' });
}
