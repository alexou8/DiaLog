import { defineConfig } from 'vitest/config';
import path from 'node:path';

/**
 * Real-database integration + security suite. Not part of `npm test` — it
 * needs a live Postgres instance (see tests/integration/README.md) and runs
 * strictly sequentially (fileParallelism: false, single fork) because every
 * test file shares one physical database and must not interleave with
 * another file's setup/teardown.
 *
 * Run with:
 *   npx vitest run --config vitest.integration.config.ts
 */
export default defineConfig({
  resolve: { alias: { '@': path.resolve(__dirname, '.') } },
  test: {
    environment: 'node',
    include: ['tests/integration/**/*.test.ts'],
    globals: false,
    fileParallelism: false,
    testTimeout: 30_000,
    hookTimeout: 30_000,
    pool: 'forks',
    poolOptions: { forks: { singleFork: true } },
    setupFiles: ['./tests/integration/setup-env.ts'],
  },
});
