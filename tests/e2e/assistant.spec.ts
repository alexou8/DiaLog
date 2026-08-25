import { test, expect } from '@playwright/test';
import { DEMO_EMAIL, DEMO_PASSWORD, signIn, signUpFreshUser } from './helpers';

test.describe('assistant', () => {
  test('demo account: a question gets an answer with a confidence badge and evidence disclosure', async ({
    page,
  }) => {
    await signIn(page, DEMO_EMAIL, DEMO_PASSWORD);
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

  test('fresh empty account: assistant says there is not enough data rather than inventing an answer', async ({
    page,
  }) => {
    await signUpFreshUser(page, { label: 'assistant' });
    await page.goto('/app/assistant');

    await page.getByLabel('Your question').fill('Why were my readings higher this week?');
    await page.getByRole('button', { name: 'Ask' }).click();

    await expect(page.getByText('You asked')).toBeVisible();
    await expect(page.getByText('Not enough evidence')).toBeVisible();
    await expect(page.getByText('Limited data')).toBeVisible();

    const disclosure = page.getByText('What was this answer based on?');
    await disclosure.click();
    await expect(
      page.getByText('No specific finding was strong enough to cite, which is why the answer is cautious.'),
    ).toBeVisible();
  });
});
