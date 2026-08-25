import { test, expect } from '@playwright/test';
import { ASSISTANT_FRESH_STATE, DEMO_STATE } from './setup/auth-state';

test.describe('assistant', () => {
  test.describe('demo account', () => {
    test.use({ storageState: DEMO_STATE });

    test('a question gets an answer with a confidence badge and evidence disclosure', async ({
      page,
    }) => {
      await page.goto('/app/assistant');

      await page.getByLabel('Your question').fill('What patterns have you noticed recently?');
      await page.getByRole('button', { name: 'Ask' }).click();

      await expect(page.getByText('You asked')).toBeVisible();
      await expect(page.getByText('What patterns have you noticed recently?')).toBeVisible();

      const confidenceBadge = page.getByText(/confidence|Not enough evidence/i).first();
      await expect(confidenceBadge).toBeVisible();

      const disclosure = page.getByText('What was this answer based on?');
      await expect(disclosure).toBeVisible();
      await disclosure.click();
      await expect(page.getByText(/Answer produced by:/)).toBeVisible();
    });
  });

  test.describe('fresh empty account', () => {
    test.use({ storageState: ASSISTANT_FRESH_STATE });

    test('says there is not enough data rather than inventing an answer', async ({ page }) => {
      await page.goto('/app/assistant');

      await page.getByLabel('Your question').fill('Why were my readings higher this week?');
      await page.getByRole('button', { name: 'Ask' }).click();

      await expect(page.getByText('You asked')).toBeVisible();
      await expect(page.getByText('Not enough evidence')).toBeVisible();
      await expect(page.getByText('Limited data')).toBeVisible();

      const disclosure = page.getByText('What was this answer based on?');
      await disclosure.click();
      await expect(
        page.getByText(
          'No specific finding was strong enough to cite, which is why the answer is cautious.',
        ),
      ).toBeVisible();
    });
  });
});
