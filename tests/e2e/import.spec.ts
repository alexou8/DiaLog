import path from 'node:path';
import { test, expect } from '@playwright/test';
import { IMPORT_STATE } from './setup/auth-state';

const SAMPLE_LOGS = path.join(process.cwd(), 'ml/data/sample_logs.csv');
const MALFORMED = path.join(process.cwd(), 'tests/e2e/fixtures/malformed.csv');

/** ml/data/sample_logs.csv contains 5 glucose rows (plus meals and medications). */
const GLUCOSE_ROWS_IN_SAMPLE = 5;

test.describe('import', () => {
  test.use({ storageState: IMPORT_STATE });

  test('review stage shows counts and saves nothing until confirmed; re-import reports duplicates', async ({
    page,
  }) => {
    // This test writes records, so it is not naturally repeatable: a retry
    // would start against an account holding the previous attempt's rows.
    // Undoing every existing batch first makes each attempt start from the
    // same state, and exercises the undo path on the way. It lives here rather
    // than in a beforeEach because the sibling test shares this account and
    // must not have its state cleared mid-run.
    await page.goto('/app/import');
    const undoDisclosures = page.getByRole('group', { name: 'Undo this import' });
    for (let remaining = await undoDisclosures.count(); remaining > 0; remaining -= 1) {
      const disclosure = undoDisclosures.first();
      await disclosure.getByText('Undo this import').click();
      await disclosure.getByRole('button', { name: 'Yes, remove these records' }).click();
      await expect(page.getByText('Your imports')).toBeVisible();
    }
    await expect(page.getByText('No imports yet')).toBeVisible();

    await page.getByLabel(/Choose a file/).setInputFiles(SAMPLE_LOGS);
    await page.getByRole('button', { name: 'Check this file' }).click();

    await expect(page.getByText('Here is what DiaLog found')).toBeVisible();
    await expect(page.getByText('Rows in the file')).toBeVisible();
    const readyToImportButton = page.getByRole('button', { name: /^Import \d+ record/ });
    await expect(readyToImportButton).toBeVisible();
    await expect(page.getByText('Nothing has been saved yet.')).toBeVisible();

    // Nothing saved yet. History is checked rather than the glucose page,
    // because the glucose page shows a rolling 30-day window and this sample
    // file is dated well outside it — an empty glucose page would prove
    // nothing either way.
    await page.goto('/app/history?type=glucose');
    await expect(page.getByText('No glucose records yet')).toBeVisible();

    // Go back, redo the analysis (file input is not preserved across navigation),
    // and confirm the import this time.
    await page.goto('/app/import');
    await page.getByLabel(/Choose a file/).setInputFiles(SAMPLE_LOGS);
    await page.getByRole('button', { name: 'Check this file' }).click();
    await expect(page.getByRole('button', { name: /^Import \d+ record/ })).toBeVisible();
    const importCountMatch = await page
      .getByRole('button', { name: /^Import \d+ record/ })
      .textContent();
    const expectedCount = Number(importCountMatch?.match(/\d+/)?.[0] ?? '0');
    expect(expectedCount).toBeGreaterThan(0);

    await page.getByRole('button', { name: /^Import \d+ record/ }).click();
    await expect(page.getByText('Import complete')).toBeVisible();
    await expect(page.getByText('Your readings are now in DiaLog')).toBeVisible();

    await page.goto('/app/history?type=glucose');
    await expect(page.getByText('No glucose records yet')).toHaveCount(0);
    await expect(page.getByText(/Imported from sample_logs\.csv/i).first()).toBeVisible();

    // Upload the same file again: everything should now be reported as duplicate.
    await page.goto('/app/import');
    await page.getByLabel(/Choose a file/).setInputFiles(SAMPLE_LOGS);
    await page.getByRole('button', { name: 'Check this file' }).click();
    await expect(page.getByText('Here is what DiaLog found')).toBeVisible();
    await expect(page.getByText('Already in DiaLog, so skipped').locator('..')).toContainText(
      String(expectedCount),
    );

    // Re-importing changed nothing: the stored record count is exactly what
    // the first import wrote.
    expect(await countGlucoseRows(page)).toBe(GLUCOSE_ROWS_IN_SAMPLE);
  });

  test('uploading a malformed file shows a clear, non-crashing error', async ({ page }) => {
    await page.goto('/app/import');
    await page.getByLabel(/Choose a file/).setInputFiles(MALFORMED);
    await page.getByRole('button', { name: 'Check this file' }).click();

    // DiaLog can't match the file to any known layout and says so in plain
    // language, rather than crashing or silently reporting zero rows. (A
    // file DiaLog *can* parse but where every row is individually invalid
    // would instead show a "rows could not be read" summary — this fixture
    // fails earlier, at layout detection.)
    await expect(
      page.getByText('DiaLog could not recognise the layout of that file.'),
    ).toBeVisible();

    // The app shell must still be intact — no crash page.
    await expect(page.getByRole('link', { name: 'DiaLog' })).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Import your data' })).toBeVisible();
  });
});

async function countGlucoseRows(page: import('@playwright/test').Page): Promise<number> {
  await page.goto('/app/history?type=glucose');
  // Scoped to the record list: `main ul > li` would also match the record-type
  // tab bar, which is itself a list.
  return page.locator('main ul > li').filter({ has: page.getByRole('group') }).count();
}
