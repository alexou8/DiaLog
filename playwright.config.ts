import { defineConfig, devices } from '@playwright/test';

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
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 2 : undefined,
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
    // The `playwright` package resolves a specific bundled Chromium build
    // (1187) that does not match the browser actually installed in this
    // container (1194), so PLAYWRIGHT_BROWSERS_PATH alone is not enough —
    // point directly at the installed binary.
    launchOptions: {
      executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome',
    },
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
