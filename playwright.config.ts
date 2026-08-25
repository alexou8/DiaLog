import { existsSync } from 'node:fs';
import { defineConfig, devices } from '@playwright/test';

/**
 * Resolve a Chromium binary to launch.
 *
 * On CI (and any normal machine) `npx playwright install chromium` puts the
 * matching build where Playwright expects it, so the right answer is to say
 * nothing and let Playwright resolve it. Some preconfigured containers ship a
 * Chromium whose build number does not match the one the installed
 * `playwright` package expects, and there `PLAYWRIGHT_BROWSERS_PATH` alone is
 * not enough — so an explicit path is used, but only when it actually exists.
 *
 * Set `PLAYWRIGHT_CHROMIUM_PATH` to override.
 */
function resolveChromium(): string | undefined {
  const configured = process.env.PLAYWRIGHT_CHROMIUM_PATH;
  if (configured && existsSync(configured)) return configured;
  const preinstalled = '/opt/pw-browsers/chromium';
  if (existsSync(preinstalled)) return preinstalled;
  return undefined;
}

const chromiumPath = resolveChromium();

/**
 * Playwright configuration for DiaLog's end-to-end and accessibility suite.
 *
 * The app is built and started on port 3100 (port 3000 is occupied by
 * something else in this environment). `reuseExistingServer: false` so every
 * run starts from a known-good server, and the timeout is generous because
 * `next build` runs first.
 */
export default defineConfig({
  testDir: './tests/e2e',
  // Resets the database and re-seeds the demo account, so a run never inherits
  // the previous run's records. See tests/e2e/global-setup.ts.
  globalSetup: './tests/e2e/global-setup.ts',
  // Serialized deliberately. The specs share a small pool of accounts (sign-up
  // is rate limited, so a fresh account per test is not available), and several
  // of them assert on record counts. Run in parallel, one spec's writes land
  // between another's count and its assertion. Sequential execution costs about
  // a minute and removes that entire class of flake.
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: 1,
  timeout: 60_000,
  expect: {
    timeout: 10_000,
  },
  reporter: process.env.CI ? [['list'], ['html', { open: 'never' }]] : 'list',
  use: {
    baseURL: 'http://localhost:3100',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    video: 'off',
    actionTimeout: 15_000,
    navigationTimeout: 30_000,
    launchOptions: chromiumPath ? { executablePath: chromiumPath } : {},
  },
  projects: [
    {
      // Runs once, before every other project, to sign in / register the
      // handful of accounts the rest of the suite reuses via storageState.
      // See tests/e2e/setup/auth.setup.ts for why this exists.
      name: 'setup',
      testMatch: /.*\.setup\.ts/,
    },
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
      testIgnore: [/keyboard-and-mobile\.spec\.ts/, /.*\.setup\.ts/],
      dependencies: ['setup'],
    },
    {
      name: 'mobile-chromium',
      // The spec itself sets an explicit viewport per test.describe block
      // (a full-size desktop viewport for the keyboard journey, and two
      // phone-sized viewports for the mobile checks), so the project's own
      // default viewport below is only a fallback.
      use: { ...devices['Pixel 7'] },
      testMatch: /keyboard-and-mobile\.spec\.ts/,
      dependencies: ['setup'],
    },
  ],
  webServer: {
    command: 'npm run build && npx next start -p 3100',
    url: 'http://localhost:3100',
    reuseExistingServer: false,
    timeout: 300_000,
    stdout: 'pipe',
    stderr: 'pipe',
  },
});
