import { test, expect } from '@playwright/test';
import { ASSISTANT_FRESH_STATE, DEMO_STATE } from './setup/auth-state';

test.describe('assistant', () => {
  test.describe('demo account', () => {
    test.use({ storageState: DEMO_STATE });

    test('a question gets an answer with a confidence badge and evidence disclosure', async ({
      page,
    }) => {
      await page.goto('/app/assistant');
      await page.waitForLoadState('networkidle');

      // A custom question, not one of the pre-set suggestion pills, so the
      // echoed question text below doesn't collide with a suggestion button
      // showing the same words.
      const question = 'How has my glucose been trending overall?';
      await page.getByLabel('Your question').fill(question);
      await page.getByRole('button', { name: 'Ask', exact: true }).click();

      await expect(page.getByText('You asked')).toBeVisible();
      await expect(page.getByText(question, { exact: true })).toBeVisible();

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
      await page.waitForLoadState('networkidle');

      // With zero logged records the app gates the question form entirely
      // (AssistantPage checks `hasData = counts.glucose > 0`) rather than
      // letting a question through to the AI pipeline — no form to submit,
      // just a plain-language explanation and a way to add data.
      await expect(page.getByText('There is nothing to ask about yet')).toBeVisible();
      await expect(
        page.getByText('The assistant can only answer from your own records.'),
      ).toBeVisible();
      await expect(page.getByLabel('Your question')).toHaveCount(0);
      await expect(page.getByRole('link', { name: 'Add a reading' })).toBeVisible();
    });
  });
});
