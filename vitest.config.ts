import { defineConfig } from 'vitest/config';
import path from 'node:path';

// `npm test` (plain `vitest run`) runs only the DB-free unit suite, so it
// works in any environment without a Postgres instance. The real-database
// integration/security suite lives in tests/integration/** and is run
// separately via vitest.integration.config.ts — see the `test:integration`
// command documented there and in tests/integration/README.md.
export default defineConfig({
  resolve: { alias: { '@': path.resolve(__dirname, '.') } },
  test: {
    environment: 'node',
    include: ['tests/unit/**/*.test.ts'],
    globals: false,
  },
});
