import { expect, test, type Page } from '@playwright/test';
import {
  DEMO_EMAIL,
  DEMO_PASSWORD,
  TEST_PASSWORD,
  completeOnboarding,
  signIn,
  signOut,
  uniqueEmail,
} from './helpers';

/**
 * The Google sign-in journeys, driven against the stand-in issuer in
 * tests/e2e/fake-google.ts. The app is exercised unmodified: PKCE, state,
 * nonce and the RS256 signature check all run for real.
 *
 * The cases that matter most here are the ones a unit test cannot reach — what
 * a person actually sees when their email is already taken, when Google will
 * not vouch for their address, and when Google is their only way in.
 *
 * Password accounts are never registered here: sign-up is rate limited to five
 * per hour per address (RATE_LIMITS.signUp) and the setup project already
 * spends most of that budget, so the cases needing an existing password account
 * borrow the seeded demo one. Google sign-ups are not rate limited by that
 * bucket, so accounts arriving through Google are created freely.
 */

const FAKE_GOOGLE = 'http://127.0.0.1:3210';

/** Choose who the stand-in Google asserts on the next sign-in. */
async function asGoogleUser(
  page: Page,
  identity: { sub: string; email: string; emailVerified?: boolean; name?: string | null },
): Promise<void> {
  const response = await page.request.post(`${FAKE_GOOGLE}/_identity`, { data: identity });
  expect(response.ok()).toBe(true);
}

/**
 * The page's own error banner. Scoped to the paragraph because Next renders a
 * permanently empty `role="alert"` route announcer on every page.
 */
const notice = (page: Page) => page.locator('p[role="alert"]');

const googleEmail = (label: string) => uniqueEmail(label).replace('@dialog.test', '@gmail.test');

test.describe('Google sign-in', () => {
  test('creates an account, then signs the same person back in', async ({ page }) => {
    const email = googleEmail('google.new');
    const sub = `sub-new-${Date.now()}`;
    await asGoogleUser(page, { sub, email, name: 'Grace Hopper' });

    await page.goto('/sign-up');
    await page.getByRole('link', { name: 'Continue with Google' }).click();

    // A brand-new person lands in onboarding, not on the dashboard.
    await expect(page).toHaveURL(/\/app\/onboarding/);
    await completeOnboarding(page);
    await signOut(page);

    // Second time through, the same Google subject is recognised and skips
    // onboarding.
    await asGoogleUser(page, { sub, email });
    await page.goto('/sign-in');
    await page.getByRole('link', { name: 'Sign in with Google' }).click();
    await expect(page).toHaveURL(/\/app$/);
  });

  test('still recognises the account after the Google address changes', async ({ page }) => {
    const sub = `sub-rename-${Date.now()}`;
    await asGoogleUser(page, { sub, email: googleEmail('google.before') });

    await page.goto('/sign-in');
    await page.getByRole('link', { name: 'Sign in with Google' }).click();
    await expect(page).toHaveURL(/\/app\/onboarding/);
    await completeOnboarding(page);
    await signOut(page);

    // Same person, new address at Google. Identity is keyed on the subject, so
    // this must be the same DiaLog account rather than a second one.
    await asGoogleUser(page, { sub, email: googleEmail('google.after') });
    await page.goto('/sign-in');
    await page.getByRole('link', { name: 'Sign in with Google' }).click();
    await expect(page).toHaveURL(/\/app$/);
  });

  test('refuses to take over an existing password account with the same email', async ({
    page,
  }) => {
    // The seeded demo account already exists, with a password.
    await asGoogleUser(page, { sub: `sub-collision-${Date.now()}`, email: DEMO_EMAIL });
    await page.goto('/sign-in');
    await page.getByRole('link', { name: 'Sign in with Google' }).click();

    // Not signed in. Told why, and told what to do instead.
    await expect(page).toHaveURL(/\/sign-in\?/);
    await expect(notice(page)).toContainText('already have a DiaLog account with that email');
    await expect(notice(page)).toContainText('connect Google from Settings');
    // The email is prefilled so they can go straight to their password.
    await expect(page.getByLabel('Email address')).toHaveValue(DEMO_EMAIL);

    // The password still works, and the account is untouched.
    await page.getByLabel('Password').fill(DEMO_PASSWORD);
    await page.getByRole('button', { name: 'Sign in' }).click();
    await expect(page).toHaveURL(/\/app$/);
  });

  test('rejects a Google account whose email is not verified', async ({ page }) => {
    await asGoogleUser(page, {
      sub: `sub-unverified-${Date.now()}`,
      email: googleEmail('google.unverified'),
      emailVerified: false,
    });

    await page.goto('/sign-in');
    await page.getByRole('link', { name: 'Sign in with Google' }).click();

    await expect(page).toHaveURL(/\/sign-in\?error=unverified_email/);
    await expect(notice(page)).toContainText('has not verified the email address');
  });

  test('rejects a callback that this browser never started', async ({ page }) => {
    // No attempt cookie, so state cannot match: a forged or replayed callback.
    await page.goto('/api/auth/google/callback?code=forged&state=forged');
    await expect(page).toHaveURL(/\/sign-in\?error=invalid_state/);
    await expect(notice(page)).toContainText('expired or did not come from this browser');
  });
});

test.describe('connecting and disconnecting Google', () => {
  test('a password account can connect Google and then sign in with it', async ({ page }) => {
    await signIn(page, DEMO_EMAIL, DEMO_PASSWORD);

    const sub = `sub-link-${Date.now()}`;
    const email = googleEmail('google.link');
    await asGoogleUser(page, { sub, email });

    await page.goto('/app/settings');
    await page.getByRole('link', { name: 'Connect Google' }).click();

    await expect(page).toHaveURL(/\/app\/settings\?linked=google/);
    await expect(page.getByText('Google account is now connected')).toBeVisible();

    // And that connection actually signs them in afterwards. Sign out from the
    // dashboard — Settings also carries a "Sign out everywhere else" button.
    await page.goto('/app');
    await signOut(page);
    await asGoogleUser(page, { sub, email });
    await page.goto('/sign-in');
    await page.getByRole('link', { name: 'Sign in with Google' }).click();
    await expect(page).toHaveURL(/\/app$/);

    // Disconnecting is allowed, because a password remains — and this leaves
    // the shared demo account exactly as the spec found it.
    await page.goto('/app/settings');
    await page.getByRole('button', { name: 'Disconnect Google' }).click();
    await expect(page).toHaveURL(/\/app\/settings\?unlinked=google/);
    await expect(page.getByText('Google has been disconnected')).toBeVisible();
    await page.goto('/app');
    await signOut(page);
    await signIn(page, DEMO_EMAIL, DEMO_PASSWORD);
  });

  test('will not disconnect Google while it is the only way in', async ({ page }) => {
    const sub = `sub-only-${Date.now()}`;
    await asGoogleUser(page, { sub, email: googleEmail('google.only') });
    await page.goto('/sign-in');
    await page.getByRole('link', { name: 'Sign in with Google' }).click();
    await completeOnboarding(page);

    await page.goto('/app/settings');
    // No password was ever set, so Settings offers to set one rather than change one.
    await expect(page.getByRole('heading', { name: 'Set a password' })).toBeVisible();

    await page.getByRole('button', { name: 'Disconnect Google' }).click();
    await expect(page.getByText('only way to sign in to this account')).toBeVisible();
  });

  test('a Google-only account can set a password and then disconnect Google', async ({ page }) => {
    const sub = `sub-setpw-${Date.now()}`;
    await asGoogleUser(page, { sub, email: googleEmail('google.setpw') });
    await page.goto('/sign-in');
    await page.getByRole('link', { name: 'Sign in with Google' }).click();
    await completeOnboarding(page);

    await page.goto('/app/settings');
    await page.locator('input[name="newPassword"]').fill(TEST_PASSWORD);
    await page.locator('input[name="confirmPassword"]').fill(TEST_PASSWORD);
    await page.getByRole('button', { name: 'Set password' }).click();
    await expect(page.getByText('Your password has been set')).toBeVisible();

    // Reload before disconnecting: setting the password revalidates the page,
    // and clicking into a section that is mid-refresh loses the click.
    await page.reload();
    await expect(page.getByRole('heading', { name: 'Change your password' })).toBeVisible();

    // With a password in place, Google is no longer the only way in.
    await page.getByRole('button', { name: 'Disconnect Google' }).click();
    await expect(page).toHaveURL(/\/app\/settings\?unlinked=google/);
    await expect(page.getByText('Google has been disconnected')).toBeVisible();
  });
});
