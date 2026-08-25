import { test as setup, expect, type Page } from '@playwright/test';
import { PrismaClient } from '@prisma/client';
import { DEMO_EMAIL, DEMO_PASSWORD, TEST_PASSWORD, uniqueEmail } from '../helpers';
import { ASSISTANT_FRESH_STATE, DEMO_STATE, IMPORT_STATE, SHARED_STATE } from './auth-state';
import { SECURITY_LABELS, provisionSecurityAccount } from './security-accounts';

/**
 * One-time authentication setup, run once per full suite invocation by the
 * `setup` Playwright project (see playwright.config.ts). It produces a small
 * number of reusable `storageState` files under `tests/e2e/.auth/` so the
 * rest of the suite can sign in for free by loading a storage state instead
 * of driving the sign-in/sign-up forms in every test.
 *
 * This matters because both auth actions are rate-limited server-side
 * (lib/auth/rate-limit.ts): sign-up to 5 attempts/hour and sign-in to
 * 10 attempts/15 minutes, *per server process* for the whole test run. If
 * every test signed in or signed up independently the suite would trip
 * those limits. Instead:
 *   - the demo account is signed in to ONCE (`demo.json`) and reused by every
 *     spec that only needs to read the seeded demo data;
 *   - a small, fixed number of fresh accounts are registered ONCE here and
 *     reused by the specs that need an isolated, writable account, instead of
 *     each test registering its own.
 * auth-and-onboarding.spec.ts is the exception: it deliberately drives the
 * real sign-up/sign-in forms itself, because exercising that UI flow is the
 * point of that spec.
 *
 * Those two, plus the three below, spend the sign-up budget exactly. Anything
 * else needing an account must be provisioned straight from the database — see
 * `security-accounts.ts`, which is what the last setup step here uses.
 */

async function signUpAndOnboard(page: Page, label: string, statePath: string): Promise<void> {
  const email = uniqueEmail(label);
  await page.goto('/sign-up');
  await page.getByLabel('Your name').fill('Test User');
  await page.getByLabel('Email address').fill(email);
  await page.getByLabel('Password', { exact: false }).fill(TEST_PASSWORD);
  await page.getByRole('button', { name: 'Create account' }).click();

  await expect(page).toHaveURL(/\/app\/onboarding/);
  // mg/dL so that plain integer test values (101, 133, 142, ...) are valid
  // readings — the onboarding default is mmol/L, which those would fail.
  await page.getByRole('radio', { name: /mg\/dL/ }).check();
  await page.getByRole('button', { name: 'Finish setup' }).click();
  await expect(page).toHaveURL(/\/app$/);

  await page.context().storageState({ path: statePath });
}

setup('authenticate as the seeded demo account', async ({ page }) => {
  await page.goto('/sign-in');
  await page.getByLabel('Email address').fill(DEMO_EMAIL);
  await page.getByLabel('Password').fill(DEMO_PASSWORD);
  await page.getByRole('button', { name: 'Sign in' }).click();
  await expect(page).toHaveURL(/\/app/);
  await page.context().storageState({ path: DEMO_STATE });
});

setup('register the shared fresh account', async ({ page }) => {
  await signUpAndOnboard(page, 'shared', SHARED_STATE);
});

setup('register the import spec fresh account', async ({ page }) => {
  await signUpAndOnboard(page, 'import', IMPORT_STATE);
});

setup('register the assistant spec fresh account', async ({ page }) => {
  await signUpAndOnboard(page, 'assistant-fresh', ASSISTANT_FRESH_STATE);
});

/**
 * account-security.spec.ts needs a disposable account per test, because every
 * one of its tests revokes its own session on purpose. Created directly in the
 * database so they cost nothing from the sign-up budget spent above.
 */
setup('provision the account-security accounts', async ({ baseURL }) => {
  const prisma = new PrismaClient();
  try {
    for (const label of SECURITY_LABELS) {
      await provisionSecurityAccount(prisma, label, baseURL as string);
    }
  } finally {
    await prisma.$disconnect();
  }
});
