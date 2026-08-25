import { test, expect } from '@playwright/test';
import { SHARED_STATE } from './setup/auth-state';

test.describe('logging records', () => {
  test.use({ storageState: SHARED_STATE });
  // All five tests write to the same shared account (kept small to conserve
  // the server's sign-up rate limit — see setup/auth.setup.ts). Serial mode
  // avoids two of them mutating/re-rendering that account's pages at once,
  // which could otherwise race the delete test's own list re-render.
  test.describe.configure({ mode: 'serial' });

  test('add a glucose reading and see it with value and status', async ({ page }) => {
    await page.goto('/app/glucose/new');
    await page.getByLabel(/Your reading/).fill('142');
    await page.getByRole('radio', { name: 'After a meal', exact: false }).check();
    await page.getByRole('button', { name: 'Save reading' }).click();

    await expect(page).toHaveURL(/\/app\/glucose\?added=1/);
    await expect(page.getByText('Reading saved')).toBeVisible();
    await expect(page.getByText('142').first()).toBeVisible();
    // A plain-language status label ("Above your target range", etc.) accompanies the value.
    await expect(page.getByText(/target range/i).first()).toBeVisible();
  });

  test('log a meal and see it in history with provenance', async ({ page }) => {
    await page.goto('/app/meals/new');
    await page.getByLabel('What did you eat?').fill('Grilled chicken and rice');
    await page.getByRole('button', { name: 'Save meal' }).click();

    await expect(page).toHaveURL(/\/app\/meals\?added=1/);
    await expect(page.getByText('Grilled chicken and rice')).toBeVisible();

    await page.goto('/app/history?type=meal');
    await expect(page.getByText('Grilled chicken and rice')).toBeVisible();
  });

  test('log an exercise session and see it in history', async ({ page }) => {
    await page.goto('/app/activity/new');
    await page.getByLabel('Activity').fill('Walking');
    await page.getByLabel('Duration').fill('30');
    await page.getByRole('button', { name: 'Save activity' }).click();

    await expect(page).toHaveURL(/\/app\/activity\?added=1/);
    await expect(page.getByText('Walking').first()).toBeVisible();

    await page.goto('/app/history?type=exercise');
    await expect(page.getByText('Walking').first()).toBeVisible();
  });

  test('log sleep and see it in history', async ({ page }) => {
    await page.goto('/app/health/sleep/new');
    // Wake time must be after bedtime for a plausible duration; defaults are
    // both "now", so push the wake time forward.
    const bedtime = await page.getByLabel('Bedtime').inputValue();
    const wakeDate = new Date(bedtime);
    wakeDate.setHours(wakeDate.getHours() + 7);
    const pad = (n: number) => String(n).padStart(2, '0');
    const wake = `${wakeDate.getFullYear()}-${pad(wakeDate.getMonth() + 1)}-${pad(wakeDate.getDate())}T${pad(wakeDate.getHours())}:${pad(wakeDate.getMinutes())}`;
    await page.getByLabel('Wake time').fill(wake);
    await page.getByRole('radio', { name: /Good/ }).check();
    await page.getByRole('button', { name: 'Save sleep' }).click();

    await expect(page).toHaveURL(/\/app\/health\?added=sleep/);
    await expect(page.getByText('Sleep saved.')).toBeVisible();
    await expect(page.getByText(/7h 0m/)).toBeVisible();

    await page.goto('/app/history?type=sleep');
    await expect(page.getByText(/7h 0m/)).toBeVisible();
  });

  test('delete a record via the two-step confirm', async ({ page }) => {
    await page.goto('/app/glucose/new');
    await page.getByLabel(/Your reading/).fill('101');
    await page.getByRole('button', { name: 'Save reading' }).click();
    await expect(page).toHaveURL(/\/app\/glucose\?added=1/);

    await page.goto('/app/history?type=glucose');
    const row = page.locator('li').filter({ hasText: '101' }).first();
    await expect(row).toBeVisible();

    // First click opens the confirmation; the record must not vanish yet.
    // <summary> exposes an ARIA role of "DisclosureTriangle" in Chromium, not
    // "button", so it is targeted by its visible text rather than by role.
    await row.locator('summary').filter({ hasText: 'Delete' }).click();
    // Scoped to this row: every row carries its own (usually-hidden) confirm
    // text, so matching page-wide could resolve to more than one row's copy.
    await expect(row.getByText('Delete this record? This cannot be undone.')).toBeVisible();
    await expect(row).toBeVisible();

    await page.getByRole('button', { name: 'Yes, delete' }).click();
    await expect(page.locator('li').filter({ hasText: '101' })).toHaveCount(0);
  });
});
