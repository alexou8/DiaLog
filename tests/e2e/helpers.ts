import type { Page } from '@playwright/test';
import { expect } from '@playwright/test';

/** Generates an email that is unique to this run so tests never collide. */
export function uniqueEmail(label: string): string {
  const stamp = `${Date.now()}-${Math.floor(Math.random() * 1e6)}`;
  return `e2e.${label}.${stamp}@dialog.test`;
}

export const DEMO_EMAIL = 'demo@dialog.health';
export const DEMO_PASSWORD = 'demo-account-2026';

/** A password that satisfies DiaLog's policy (at least 10 characters). */
export const TEST_PASSWORD = 'correct-horse-battery-2026';

/**
 * Registers a brand-new account with a unique email and completes onboarding
 * with sensible defaults, landing on the dashboard. Returns the email used so
 * callers can sign back in later if needed.
 */
export async function signUpFreshUser(
  page: Page,
  options?: { label?: string; displayName?: string },
): Promise<{ email: string; password: string }> {
  const email = uniqueEmail(options?.label ?? 'user');
  const password = TEST_PASSWORD;
  const displayName = options?.displayName ?? 'Test User';

  await page.goto('/sign-up');
  await page.getByLabel('Your name').fill(displayName);
  await page.getByLabel('Email address').fill(email);
  await page.getByLabel('Password', { exact: false }).fill(password);
  await page.getByRole('button', { name: 'Create account' }).click();

  await expect(page).toHaveURL(/\/app\/onboarding/);
  await completeOnboarding(page);

  return { email, password };
}

/** Fills and submits the onboarding form with defaults, landing on /app. */
export async function completeOnboarding(page: Page): Promise<void> {
  await expect(page.getByRole('heading', { level: 1 })).toBeVisible();
  await page.getByRole('button', { name: 'Finish setup' }).click();
  await expect(page).toHaveURL(/\/app$/);
}

/** Signs in an existing account and waits for the app shell to load. */
export async function signIn(page: Page, email: string, password: string): Promise<void> {
  await page.goto('/sign-in');
  await page.getByLabel('Email address').fill(email);
  await page.getByLabel('Password').fill(password);
  await page.getByRole('button', { name: 'Sign in' }).click();
  await expect(page).toHaveURL(/\/app/);
}

export async function signOut(page: Page): Promise<void> {
  await page.getByRole('button', { name: 'Sign out' }).click();
  await expect(page).toHaveURL('/');
}
