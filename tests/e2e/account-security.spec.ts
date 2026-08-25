import { expect, test, type Browser, type Page } from '@playwright/test';
import { TEST_PASSWORD } from './helpers';
import { securityState, type SecurityLabel } from './setup/security-accounts';

/**
 * The account-security forms on Settings: changing a password, and signing
 * other devices out.
 *
 * Both revoke the session cookie the request arrives with, and both used to be
 * Server Actions rendered through `useActionState`. That shape fails
 * intermittently — measured at 6 failures in 10 for the change-password form —
 * because the client router has other requests for the same document in flight
 * carrying the cookie the mutation just invalidated, and a redirect to
 * /sign-in on any of them makes the router discard the action's result. They
 * are now plain HTML forms posting to route handlers, so the browser owns the
 * navigation and nothing holding the old cookie outlives it. See
 * lib/auth/route-form.ts.
 *
 * Every test here gets its own account, provisioned straight from the database
 * by auth.setup.ts, because each one deliberately revokes its own session and
 * because the suite's sign-up budget is already fully spent — see
 * setup/security-accounts.ts.
 */

/** Opens a page already signed in as one of the provisioned accounts. */
async function asAccount(browser: Browser, label: SecurityLabel): Promise<Page> {
  const context = await browser.newContext({ storageState: securityState(label) });
  return context.newPage();
}

async function changePassword(page: Page, current: string, next: string): Promise<void> {
  await page.locator('input[name="currentPassword"]').fill(current);
  await page.locator('input[name="newPassword"]').fill(next);
  await page.locator('input[name="confirmPassword"]').fill(next);
  await page.getByRole('button', { name: 'Change password' }).click();
}

test.describe('account security', () => {
  /**
   * The repetition is the point. A single pass proved nothing: the old
   * implementation confirmed four times in ten, so any one attempt was likely
   * enough to look healthy. Six in a row would have got through about 0.4% of
   * the time.
   */
  test('changing the password confirms every time, not four times in ten', async ({ browser }) => {
    test.setTimeout(180_000);
    const page = await asAccount(browser, 'rounds');
    let current = TEST_PASSWORD;

    for (let round = 0; round < 6; round++) {
      const next = `changed-password-round-${round}`;

      await page.goto('/app/settings');
      // A dropped confirmation used to leave the browser holding a revoked
      // cookie, so being signed out here is the same defect one round late.
      await expect(
        page.getByRole('heading', { name: 'Change your password' }),
        `round ${round}: the session did not survive the previous change`,
      ).toBeVisible();

      await changePassword(page, current, next);

      await expect(
        page.getByText('Your password has been changed'),
        `round ${round}: no confirmation`,
      ).toBeVisible();

      current = next;
    }

    // Still signed in on this device after six consecutive revocations.
    await page.goto('/app/settings');
    await expect(page.getByRole('heading', { name: 'Change your password' })).toBeVisible();
    await page.context().close();
  });

  test('a wrong current password is refused on the field it belongs to', async ({ browser }) => {
    const page = await asAccount(browser, 'refuse');
    await page.goto('/app/settings');

    await changePassword(page, 'not-my-current-password', 'a-perfectly-fine-password');

    await expect(page.getByText('That is not your current password')).toBeVisible();
    // Refusals travel as an outcome code on the query string, so the field they
    // belong to still carries them — the behaviour returned field errors gave.
    await expect(page.locator('input[name="currentPassword"]')).toHaveAttribute(
      'aria-invalid',
      'true',
    );

    // Nothing was revoked by a refusal, so this session is still good.
    await page.goto('/app/settings');
    await expect(page.getByRole('heading', { name: 'Change your password' })).toBeVisible();
    await page.context().close();
  });

  test('a password change signs other devices out but keeps this one in', async ({ browser }) => {
    const page = await asAccount(browser, 'revoke-password');
    // A second browser holding a valid session for the same account.
    const otherPage = await asAccount(browser, 'revoke-password');
    await otherPage.goto('/app/settings');
    await expect(otherPage.getByRole('heading', { name: 'Change your password' })).toBeVisible();

    await page.goto('/app/settings');
    await changePassword(page, TEST_PASSWORD, 'the-replacement-password');
    await expect(page.getByText('Your password has been changed')).toBeVisible();

    // The security property the tokenVersion bump exists for. Keeping this
    // true is what stops the fix quietly becoming a grace period.
    await otherPage.goto('/app');
    await expect(otherPage).toHaveURL(/\/sign-in/);

    // ...while the device that made the change is untouched.
    await page.goto('/app/settings');
    await expect(page.getByRole('heading', { name: 'Change your password' })).toBeVisible();

    await page.context().close();
    await otherPage.context().close();
  });

  test('sign out everywhere confirms, and ends the other session only', async ({ browser }) => {
    const page = await asAccount(browser, 'revoke-sessions');
    const otherPage = await asAccount(browser, 'revoke-sessions');
    await otherPage.goto('/app/settings');
    await expect(otherPage.getByRole('heading', { name: 'Change your password' })).toBeVisible();

    await page.goto('/app/settings');
    await page.getByRole('button', { name: 'Sign out everywhere else' }).click();
    await expect(page.getByText('Every other device has been signed out')).toBeVisible();

    await otherPage.goto('/app');
    await expect(otherPage).toHaveURL(/\/sign-in/);

    await page.goto('/app/settings');
    await expect(page.getByRole('heading', { name: 'Change your password' })).toBeVisible();

    await page.context().close();
    await otherPage.context().close();
  });
});
