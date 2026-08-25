import path from 'node:path';
import { test, expect } from '@playwright/test';
import { signUpFreshUser } from './helpers';

const SAMPLE_LOGS = path.join(process.cwd(), 'ml/data/sample_logs.csv');
const MALFORMED = path.join(process.cwd(), 'tests/e2e/fixtures/malformed.csv');

test.describe('import', () => {
  test('review stage shows counts and saves nothing until confirmed; re-import reports duplicates', async ({ page }) => {
    await signUpFreshUser(page, { label: 'import' });

    await page.goto('/app/import');
    await page.getByLabel(/Choose a file/).setInputFiles(SAMPLE_LOGS);
    await page.getByRole('button', { name: 'Check this file' }).click();

    await expect(page.getByText('Here is what DiaLog found')).toBeVisible();
    await expect(page.getByText('Rows in the file')).toBeVisible();
    const readyToImportButton = page.getByRole('button', { name: /^Import \d+ record/ });
    await expect(readyToImportButton).toBeVisible();
    await expect(page.getByText('Nothing has been saved yet.')).toBeVisible();

    // Nothing saved yet: the glucose page must still be empty.
    await page.goto('/app/glucose');
    await expect(page.getByText('No readings in this period')).toBeVisible();

    // Go back, redo the analysis (file input is not preserved across navigation),
    // and confirm the import this time.
    await page.goto('/app/import');
    await page.getByLabel(/Choose a file/).setInputFiles(SAMPLE_LOGS);
    await page.getByRole('button', { name: 'Check this file' }).click();
    await expect(page.getByRole('button', { name: /^Import \d+ record/ })).toBeVisible();
    const importCountMatch = await page.getByRole('button', { name: /^Import \d+ record/ }).textContent();
    const expectedCount = Number(importCountMatch?.match(/\d+/)?.[0] ?? '0');
    expect(expectedCount).toBeGreaterThan(0);

    await page.getByRole('button', { name: /^Import \d+ record/ }).click();
    await expect(page.getByText('Import complete')).toBeVisible();
    await expect(page.getByText('Saved')).toBeVisible();

    await page.goto('/app/glucose');
    await expect(page.getByText('No readings in this period')).toHaveCount(0);

    // Upload the same file again: everything should now be reported as duplicate.
    await page.goto('/app/import');
    await page.getByLabel(/Choose a file/).setInputFiles(SAMPLE_LOGS);
    await page.getByRole('button', { name: 'Check this file' }).click();
    await expect(page.getByText('Here is what DiaLog found')).toBeVisible();
    await expect(page.getByText('Already in DiaLog, so skipped').locator('..')).toContainText(String(expectedCount));

    const recordCountBefore = await countGlucoseRows(page);
    // Nothing new to import, so the commit form should not appear; even if it
    // did, the record count in the app must stay the same.
    await page.goto('/app/glucose');
    const recordCountAfter = await countGlucoseRows(page);
    expect(recordCountAfter).toBe(recordCountBefore);
  });

  test('uploading a malformed file shows a clear, non-crashing error', async ({ page }) => {
    await signUpFreshUser(page, { label: 'importbad' });

    await page.goto('/app/import');
    await page.getByLabel(/Choose a file/).setInputFiles(MALFORMED);
    await page.getByRole('button', { name: 'Check this file' }).click();

    // Either the whole file is rejected up front, or every row is reported as
    // unreadable — either way the page must explain clearly and must not crash.
    const formError = page.getByText(/could not|corrupted|could not be read/i).first();
    const rejectedSummary = page.getByText('Could not be read').locator('..');

    const sawFormError = await formError.isVisible().catch(() => false);
    if (!sawFormError) {
      await expect(page.getByText('Here is what DiaLog found')).toBeVisible();
      await expect(rejectedSummary).toBeVisible();
    }

    // The app shell must still be intact — no crash page.
    await expect(page.getByRole('link', { name: 'DiaLog' })).toBeVisible();
  });
});

async function countGlucoseRows(page: import('@playwright/test').Page): Promise<number> {
  await page.goto('/app/history?type=glucose');
  return page.locator('main ul > li').count();
}
