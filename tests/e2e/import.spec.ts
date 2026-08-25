import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { test, expect } from '@playwright/test';
import { countRecordRows } from './records';
import { IMPORT_STATE } from './setup/auth-state';

const MALFORMED = path.join(process.cwd(), 'tests/e2e/fixtures/malformed.csv');

/** ml/data/sample_logs.csv holds 13 records: 5 glucose, 5 meals, 3 medications. */
const GLUCOSE_ROWS_IN_SAMPLE = 5;
const TOTAL_ROWS_IN_SAMPLE = 13;

/**
 * Write a copy of the sample log whose timestamps are shifted by a unique
 * number of days.
 *
 * The import test necessarily writes records, CI retries it, and it shares an
 * account with its sibling — so importing the same fixed file every time makes
 * the second attempt a no-op (every row a duplicate) and the test unrepeatable.
 * Shifting the dates means every attempt imports genuinely new records, while
 * re-importing the *same generated file* still exercises duplicate detection.
 */
function uniqueSampleLog(): string {
  const source = readFileSync(path.join(process.cwd(), 'ml/data/sample_logs.csv'), 'utf8');
  // A day offset that differs per attempt, kept inside a plausible range.
  const offsetDays = Math.floor(Math.random() * 3000) + 1;
  const shifted = source.replace(/^(\d{4})-(\d{2})-(\d{2}) /gm, (_match, y, m, d) => {
    const shiftedDate = new Date(Date.UTC(Number(y), Number(m) - 1, Number(d)));
    shiftedDate.setUTCDate(shiftedDate.getUTCDate() - offsetDays);
    return `${shiftedDate.toISOString().slice(0, 10)} `;
  });
  const file = path.join(mkdtempSync(path.join(tmpdir(), 'dialog-import-')), 'sample_logs.csv');
  writeFileSync(file, shifted, 'utf8');
  return file;
}

test.describe('import', () => {
  test.use({ storageState: IMPORT_STATE });

  test('review stage shows counts and saves nothing until confirmed; re-import reports duplicates', async ({
    page,
  }) => {
    // Deliberately written to hold regardless of what this account already
    // contains. The test writes records, CI retries it, and it shares an
    // account with its sibling — so asserting on absolute counts or on an
    // empty starting state makes it unrepeatable. Every assertion below is
    // about the *change* the import causes.
    const sampleLog = uniqueSampleLog();
    const before = await countRecordRows(page, '/app/history?type=glucose');

    await page.goto('/app/import');
    await page.getByLabel(/Choose a file/).setInputFiles(sampleLog);
    await page.getByRole('button', { name: 'Check this file' }).click();

    await expect(page.getByText('Here is what DiaLog found')).toBeVisible();
    await expect(page.getByText('Rows in the file')).toBeVisible();
    const readyToImport = page.getByRole('button', { name: /^Import \d+ record/ });
    await expect(readyToImport).toBeVisible();
    await expect(page.getByText('Nothing has been saved yet.')).toBeVisible();

    const offered = Number((await readyToImport.textContent())?.match(/\d+/)?.[0] ?? '0');
    expect(offered).toBe(TOTAL_ROWS_IN_SAMPLE);

    // The review stage must not have written anything.
    expect(await countRecordRows(page, '/app/history?type=glucose')).toBe(before);

    // Redo the analysis (the file input does not survive navigation) and
    // confirm it this time.
    await page.goto('/app/import');
    await page.getByLabel(/Choose a file/).setInputFiles(sampleLog);
    await page.getByRole('button', { name: 'Check this file' }).click();
    const confirm = page.getByRole('button', { name: /^Import \d+ record/ });
    await expect(confirm).toBeVisible();
    const committing = Number((await confirm.textContent())?.match(/\d+/)?.[0] ?? '0');
    expect(committing).toBe(offered);

    await confirm.click();
    await expect(page.getByText('Import complete')).toBeVisible();
    await expect(page.getByText('Your readings are now in DiaLog')).toBeVisible();

    // Exactly the file's glucose rows were added, and they carry their origin.
    const afterFirstImport = await countRecordRows(page, '/app/history?type=glucose'); // navigates to history
    expect(afterFirstImport).toBe(before + GLUCOSE_ROWS_IN_SAMPLE);
    await expect(page.getByText(/Imported from sample_logs\.csv via /i).first()).toBeVisible();

    // Re-uploading the very same file reports every row as a duplicate and
    // adds nothing.
    await page.goto('/app/import');
    await page.getByLabel(/Choose a file/).setInputFiles(sampleLog);
    await page.getByRole('button', { name: 'Check this file' }).click();
    await expect(page.getByText('Here is what DiaLog found')).toBeVisible();
    await expect(page.getByText('Already in DiaLog, so skipped').locator('..')).toContainText(
      String(TOTAL_ROWS_IN_SAMPLE),
    );
    await expect(page.getByRole('button', { name: /^Import \d+ record/ })).toHaveCount(0);
    await expect(page.getByText('Nothing new to add')).toBeVisible();

    expect(await countRecordRows(page, '/app/history?type=glucose')).toBe(afterFirstImport);
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
