import { PrismaClient } from '@prisma/client';
import { announceSeed } from './unique';

/**
 * The only databases this suite may wipe.
 *
 * `globalSetup` deletes every user and audit row, so the database it points at
 * must be one that exists to be wiped. Prisma loads `.env`, and `.env.example`
 * sends `DATABASE_URL` to the developer's real `dialog` database (or, in a
 * hurry, to a hosted one) — so the suite never reads `DATABASE_URL` from the
 * environment. It uses `E2E_DATABASE_URL`, which must name `dialog_e2e` (or
 * `dialog_test`, the integration suite's database), and playwright.config.ts
 * copies that into `DATABASE_URL` for the server and every worker. The
 * integration suite has the same kind of guard (tests/integration/setup-env.ts);
 * this is its e2e counterpart.
 *
 * The check is deliberately a name check, not a host check: a disposable
 * database is a convention the developer opts into by naming it, exactly as
 * with `dialog_test`.
 */
export const FALLBACK_E2E_DB =
  'postgresql://postgres:dialog@127.0.0.1:5432/dialog_e2e?schema=public';
const DISPOSABLE_DB = /\/(dialog_e2e|dialog_test)(\?|$)/;

/** Host and database name only — a connection string carries a password. */
function describeTarget(connectionString: string): string {
  try {
    const url = new URL(connectionString);
    return `${url.host}${url.pathname}`;
  } catch {
    return '<unparseable connection string>';
  }
}

/**
 * Throws unless every connection string the suite will use names a disposable
 * database, and never when NODE_ENV is production. Returns nothing: callers
 * either proceed or the run stops before anything is deleted.
 */
export function assertDisposableE2eDatabase(
  urls: Record<string, string | undefined>,
  nodeEnv: string | undefined = process.env.NODE_ENV,
): void {
  if (nodeEnv === 'production') {
    throw new Error('Refusing to run the e2e suite with NODE_ENV=production.');
  }
  for (const [name, value] of Object.entries(urls)) {
    if (!value || !DISPOSABLE_DB.test(value)) {
      throw new Error(
        `Refusing to reset ${name} (${value ? describeTarget(value) : 'unset'}): the e2e suite deletes every account ` +
          'in its database, so that database must be named dialog_e2e (or dialog_test). ' +
          'Set E2E_DATABASE_URL to a disposable database, and migrate it with ' +
          '`DATABASE_URL=<url> DIRECT_DATABASE_URL=<url> npx prisma migrate deploy`.',
      );
    }
  }
}

/**
 * Resolve the e2e database from `E2E_DATABASE_URL` (falling back to a local
 * `dialog_e2e`), validate it, and return the pair of URLs to publish as
 * `DATABASE_URL` / `DIRECT_DATABASE_URL`. Called from playwright.config.ts so a
 * bad value stops the run before the web server or any worker starts.
 */
export function resolveE2eDatabase(env: NodeJS.ProcessEnv = process.env): {
  url: string;
  directUrl: string;
} {
  const url = env.E2E_DATABASE_URL ?? FALLBACK_E2E_DB;
  const directUrl = env.E2E_DIRECT_DATABASE_URL ?? url;
  assertDisposableE2eDatabase({ E2E_DATABASE_URL: url, E2E_DIRECT_DATABASE_URL: directUrl });
  return { url, directUrl };
}

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
 *   2. unique, seeded content (tests/e2e/unique.ts) — two tests in one run
 *      cannot collide on a dedupe key, and a failure can be replayed;
 *   3. serial execution (playwright.config.ts) — specs sharing an account
 *      cannot interleave their writes.
 */
export default async function globalSetup(): Promise<void> {
  // Re-check what Prisma will actually connect to, not just what the config
  // validated: this is the call site that deletes, so the guard sits here too.
  assertDisposableE2eDatabase({
    DATABASE_URL: process.env.DATABASE_URL,
    DIRECT_DATABASE_URL: process.env.DIRECT_DATABASE_URL,
  });

  // Published through the environment so every worker draws from the same seed
  // (see tests/e2e/unique.ts).
  announceSeed();

  const prisma = new PrismaClient();
  try {
    await prisma.user.deleteMany({});
    await prisma.auditEvent.deleteMany({});
  } finally {
    await prisma.$disconnect();
  }

  // Re-seed the demo account the read-only specs rely on.
  //
  // tsx's own entry point is run through the current Node binary rather than
  // spawning `npx`: on Windows `npx` is `npx.cmd`, which `execFileSync` cannot
  // launch without a shell (ENOENT, or EINVAL since the CVE-2024-27980
  // hardening), and a shell would reintroduce quoting problems. This works the
  // same on every platform.
  const { execFileSync } = await import('node:child_process');
  execFileSync(process.execPath, [require.resolve('tsx/cli'), 'prisma/seed.ts'], {
    stdio: 'inherit',
    env: process.env,
  });
}
