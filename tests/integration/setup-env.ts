/**
 * Vitest setup file for the integration/security suite (see
 * vitest.integration.config.ts). Runs before any test file — and before
 * `@/lib/db/prisma` is ever imported by one — so it MUST point the
 * datasource at a separate `dialog_test` database before anything touches
 * Prisma. This guarantees these tests can never write to the developer's
 * real `dialog` database, no matter what `.env` says.
 *
 * Override with TEST_DATABASE_URL / TEST_DIRECT_DATABASE_URL if the test
 * database lives somewhere other than the local default.
 */
const FALLBACK_TEST_DB = 'postgresql://postgres:dialog@127.0.0.1:5432/dialog_test?schema=public';

const testUrl = process.env.TEST_DATABASE_URL ?? FALLBACK_TEST_DB;
const testDirectUrl = process.env.TEST_DIRECT_DATABASE_URL ?? testUrl;

if (!/\/dialog_test(\?|$)/.test(testUrl)) {
  throw new Error(
    `Refusing to run integration tests against "${testUrl}" — it does not look like the dialog_test database. ` +
      'Set TEST_DATABASE_URL explicitly if you really mean to use a different name.',
  );
}

process.env.DATABASE_URL = testUrl;
process.env.DIRECT_DATABASE_URL = testDirectUrl;
process.env.AUTH_SECRET ??= 'integration-test-secret-please-do-not-use-in-prod-00000000';
