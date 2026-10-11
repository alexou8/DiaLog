import { test, expect } from '@playwright/test';
import { DEMO_STATE } from './setup/auth-state';

test.describe('dashboard and insights (demo account)', () => {
  test.use({ storageState: DEMO_STATE });

  test.beforeEach(async ({ page }) => {
    await page.goto('/app');
  });

  test('dashboard shows the latest reading, 30-day stats, and a timeline chart with a table disclosure', async ({
    page,
  }) => {
    await expect(page).toHaveURL(/\/app$/);
    await expect(page.getByText('Your most recent reading')).toBeVisible();
    await expect(page.getByText('Your last 30 days')).toBeVisible();
    await expect(page.getByText(/Average reading/)).toBeVisible();
    await expect(page.getByText(/Readings in your target range/).first()).toBeVisible();
    await expect(page.getByText(/Variability/).first()).toBeVisible();

    const chart = page.getByRole('img', { name: /Glucose readings over time/ });
    await expect(chart).toBeVisible();

    const disclosure = page.getByText('View this chart as a table');
    await expect(disclosure).toBeVisible();
    await disclosure.click();

    const table = page
      .locator('table')
      .filter({ has: page.getByRole('columnheader', { name: 'When' }) });
    await expect(table).toBeVisible();
    await expect(table.locator('tbody tr').first()).toBeVisible();
    // `not.toHaveCount(0)` auto-waits, unlike a raw locator.count().
    await expect(table.locator('tbody tr')).not.toHaveCount(0);
  });

  test('insights page shows observations with evidence badges and an expandable "why" panel', async ({
    page,
  }) => {
    await page.goto('/app/insights');
    await expect(page.getByRole('heading', { name: 'Insights' })).toBeVisible();

    const observationCards = page
      .locator('li')
      .filter({ has: page.getByText('Why am I seeing this?') });
    await expect(observationCards.first()).toBeVisible();

    const firstWhy = observationCards.first().getByText('Why am I seeing this?');
    await firstWhy.click();
    await expect(
      observationCards
        .first()
        .getByText(/Records compared|record/i)
        .first(),
    ).toBeVisible();
    await expect(observationCards.first().getByText(/\d+ records?\./)).toBeVisible();
  });

  test('reports page renders the weekly report and clinician questions', async ({ page }) => {
    await page.goto('/app/reports');
    await expect(page.getByRole('heading', { name: 'Your week' })).toBeVisible();
    await expect(page.getByText('Average reading')).toBeVisible();
    await expect(page.getByText('Readings inside your target range')).toBeVisible();
    await expect(
      page.getByRole('heading', {
        name: 'Questions you might raise with your healthcare professional',
      }),
    ).toBeVisible();
  });
  // A client-side navigation that stays inside one section (a filter that only
  // changes the query string, or the redirect after saving a form) used to
  // hang whenever a loading.tsx boundary stayed mounted across it: the URL
  // never changed and the old view stayed on screen. Each check marks the
  // document first and asserts the mark survives, so a fallback full-page load
  // cannot pass for a working in-app navigation.
  test('period and record-type filters change the view in place', async ({ page }) => {
    async function hydrate() {
      // The Add panel only opens once React owns the page; a click before
      // that would be a plain link load rather than the navigation under test.
      await page.getByRole('button', { name: 'Add' }).click();
      const panel = page.getByRole('dialog', { name: 'What would you like to log?' });
      await expect(panel).toBeVisible();
      await page.keyboard.press('Escape');
      await expect(panel).toBeHidden();
      await page.evaluate(() => {
        (window as unknown as { __sameDocument?: boolean }).__sameDocument = true;
      });
    }
    const stayedInDocument = () =>
      page.evaluate(
        () => (window as unknown as { __sameDocument?: boolean }).__sameDocument === true,
      );

    await page.goto('/app/glucose');
    await hydrate();
    const period = page.getByRole('navigation', { name: 'Time period' });
    await period.getByRole('link', { name: '7 days' }).click();
    await expect(page).toHaveURL(/\/app\/glucose\?days=7$/);
    await expect(period.getByRole('link', { name: '7 days' })).toHaveAttribute(
      'aria-current',
      'page',
    );
    await expect(
      page.getByText('Your readings for the last 7 days', { exact: false }),
    ).toBeVisible();
    expect(await stayedInDocument()).toBe(true);

    await page.goto('/app/history');
    await hydrate();
    const types = page.getByRole('navigation', { name: 'Record type' });
    await types.getByRole('link', { name: 'Meals' }).click();
    await expect(page).toHaveURL(/\/app\/history\?type=meal$/);
    await expect(types.getByRole('link', { name: 'Meals' })).toHaveAttribute(
      'aria-current',
      'page',
    );
    expect(await stayedInDocument()).toBe(true);
  });
});
