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
});
