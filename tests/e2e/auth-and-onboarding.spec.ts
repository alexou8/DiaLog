import { test, expect } from '@playwright/test';
import { DEMO_EMAIL, DEMO_PASSWORD, TEST_PASSWORD, signIn, signOut, uniqueEmail } from './helpers';

test.describe('sign up and onboarding', () => {
  test('sign up, onboard, land on dashboard, sign out, sign back in, see own data', async ({
    page,
  }) => {
    const email = uniqueEmail('onboard');

    await page.goto('/sign-up');
    await page.getByLabel('Your name').fill('Ada Lovelace');
    await page.getByLabel('Email address').fill(email);
    await page.getByLabel('Password', { exact: false }).fill(TEST_PASSWORD);
    await page.getByRole('button', { name: 'Create account' }).click();

    await expect(page).toHaveURL(/\/app\/onboarding/);

    // Fill the onboarding form explicitly (name, unit, condition, timezone, goals).
    await page.getByLabel('What should we call you?').fill('Ada');
    await page.getByRole('radio', { name: /mg\/dL/ }).check();
    await page.getByRole('radio', { name: 'Type 2 diabetes' }).check();
    await page.getByLabel('Your time zone').selectOption('America/Toronto');
    await page.getByRole('checkbox', { name: 'Understand my patterns' }).check();
    await page.getByRole('checkbox', { name: 'Steadier readings day to day' }).check();

    await page.getByRole('button', { name: 'Finish setup' }).click();

    await expect(page).toHaveURL(/\/app$/);
    await expect(page.getByRole('heading', { level: 1 })).toContainText('Ada');

    await signOut(page);
    await expect(page).toHaveURL('/');

    await signIn(page, email, TEST_PASSWORD);
    await expect(page).toHaveURL(/\/app$/);
    await expect(page.getByRole('heading', { level: 1 })).toContainText('Ada');
  });

  test('signing up with an existing email shows a helpful error', async ({ page }) => {
    await page.goto('/sign-up');
    await page.getByLabel('Email address').fill(DEMO_EMAIL);
    await page.getByLabel('Password', { exact: false }).fill(TEST_PASSWORD);
    await page.getByRole('button', { name: 'Create account' }).click();

    await expect(page.getByText(/already exists/i)).toBeVisible();
    // Must not have created a session / navigated away.
    await expect(page).toHaveURL(/\/sign-up/);
  });

  test('wrong password shows a helpful error', async ({ page }) => {
    await page.goto('/sign-in');
    await page.getByLabel('Email address').fill(DEMO_EMAIL);
    await page.getByLabel('Password').fill('definitely-the-wrong-password');
    await page.getByRole('button', { name: 'Sign in' }).click();

    await expect(page.getByText(/do not match an account/i)).toBeVisible();
    await expect(page).toHaveURL(/\/sign-in/);
  });

  test('visiting /app while signed out redirects to sign-in', async ({ page }) => {
    await page.goto('/app');
    await expect(page).toHaveURL(/\/sign-in/);
  });
});
