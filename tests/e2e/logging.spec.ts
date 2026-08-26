import { test, expect } from '@playwright/test';
import { countRecordRows, recordRows } from './records';
import { SHARED_STATE } from './setup/auth-state';
import {
  shiftLocalDateTime,
  uniqueDurationMinutes,
  uniqueGlucoseValue,
  uniqueMinuteOffset,
  uniqueText,
} from './unique';

test.describe('logging records', () => {
  test.use({ storageState: SHARED_STATE });
  // All five tests write to the same shared account (kept small to conserve
  // the server's sign-up rate limit — see setup/auth.setup.ts). Serial mode
  // avoids two of them mutating/re-rendering that account's pages at once,
  // which could otherwise race the delete test's own list re-render.
  test.describe.configure({ mode: 'serial' });

  test('add a glucose reading and see it with value and status', async ({ page }) => {
    await page.goto('/app/glucose/new');
    const label = (await page.getByText(/^Your reading \(/).textContent()) ?? '';
    const value = uniqueGlucoseValue(label);
    await page.getByLabel(/Your reading/).fill(value);
    await page.getByRole('radio', { name: 'After a meal', exact: false }).check();
    await page.getByRole('button', { name: 'Save reading' }).click();

    await expect(page).toHaveURL(/\/app\/glucose\?added=1/);
    await expect(page.getByText('Reading saved')).toBeVisible();
    // Filtered to what is actually on screen: the timeline chart labels each
    // point with an SVG <title> ("... 240 mg/dL - Above your target range"),
    // which matches both of these texts but is never visible. Whether that
    // hidden title or the list row came first in the DOM decided whether this
    // test passed, which is what made it flaky.
    await expect(page.getByText(value).filter({ visible: true }).first()).toBeVisible();
    // A plain-language status label ("Above your target range", etc.) accompanies the value.
    await expect(
      page
        .getByText(/target range/i)
        .filter({ visible: true })
        .first(),
    ).toBeVisible();
  });

  test('log a meal and see it in history with provenance', async ({ page }) => {
    const description = uniqueText('Grilled chicken and rice');
    await page.goto('/app/meals/new');
    await page.getByLabel('What did you eat?').fill(description);
    await page.getByRole('button', { name: 'Save meal' }).click();

    await expect(page).toHaveURL(/\/app\/meals\?added=1/);
    await expect(page.getByText(description)).toBeVisible();

    await page.goto('/app/history?type=meal');
    await expect(page.getByText(description)).toBeVisible();
  });

  test('log an exercise session and see it in history', async ({ page }) => {
    const activity = uniqueText('Walking');
    const duration = uniqueDurationMinutes();
    await page.goto('/app/activity/new');
    await page.getByLabel('Activity').fill(activity);
    await page.getByLabel('Duration').fill(duration);
    await page.getByRole('button', { name: 'Save activity' }).click();

    await expect(page).toHaveURL(/\/app\/activity\?added=1/);
    await expect(page.getByText(activity).first()).toBeVisible();

    await page.goto('/app/history?type=exercise');
    await expect(page.getByText(activity).first()).toBeVisible();
  });

  test('log sleep and see it in history', async ({ page }) => {
    await page.goto('/app/health/sleep/new');
    // Wake time must be after bedtime for a plausible duration; defaults are
    // both "now", so push the wake time forward.
    const bedtime = shiftLocalDateTime(
      await page.getByLabel('Bedtime').inputValue(),
      uniqueMinuteOffset(),
    );
    await page.getByLabel('Bedtime').fill(bedtime);
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
    // Asserted on the row-count delta rather than on matching row text: the
    // history list does not render notes, and filtering on a fixed value
    // ('101') matched any row merely containing those digits — so an attempt
    // that failed part-way left a matching record behind and the next run
    // found two.
    const rows = recordRows(page);
    const before = await countRecordRows(page, '/app/history?type=glucose');

    await page.goto('/app/glucose/new');
    // A value unique to this attempt. Dedupe keys are content hashes over
    // (type, timestamp-to-the-minute, value), so re-running with a fixed value
    // inside the same minute collides with the previous attempt's record and
    // the save is rejected as a duplicate. The unit is read from the field's
    // own label so the value is always in the plausible range for it.
    const readingLabel = (await page.getByText(/^Your reading \(/).textContent()) ?? '';
    const value = readingLabel.includes('mmol/L')
      ? (Math.random() * 10 + 4).toFixed(1) // 4.0-14.0 mmol/L
      : String(Math.floor(Math.random() * 180) + 80); // 80-260 mg/dL
    await page.getByLabel(/Your reading/).fill(value);
    await page.getByRole('button', { name: 'Save reading' }).click();
    await expect(page).toHaveURL(/\/app\/glucose\?added=1/);

    await page.goto('/app/history?type=glucose');
    await expect(rows).toHaveCount(before + 1);

    // Newest first, so the reading just saved is the first row.
    const row = rows.first();

    // Each row's trigger carries a per-row accessible name, so this resolves
    // to one button even though every row has a "Delete".
    const trigger = row.getByRole('button', { name: 'Delete this record' });
    await trigger.click();

    // The dialog is portalled to the end of <body>, so it is addressed on the
    // page rather than inside the row. Asserting the alertdialog role is also
    // the assertion that it is a real modal.
    const confirm = page.getByRole('alertdialog');
    await expect(confirm).toBeVisible();
    await expect(confirm.getByText('It cannot be undone.')).toBeVisible();

    // Backing out must delete nothing. This is also the assertion that the
    // rest of the page really is inert while the dialog is open: the row
    // count below can only be taken once the dialog has closed, because
    // Radix marks everything behind the modal aria-hidden and role-based
    // locators correctly refuse to match hidden content. A count taken with
    // the dialog still open resolves to zero, which is the modal working.
    await confirm.getByRole('button', { name: 'Keep it' }).click();
    await expect(confirm).toBeHidden();
    await expect(trigger).toBeFocused();
    await expect(rows).toHaveCount(before + 1);

    await trigger.click();
    await confirm.getByRole('button', { name: 'Yes, delete' }).click();
    // Reload before asserting: the claim under test is that the record is gone
    // from the server, and a reload proves that rather than the client router
    // cache's view of the list.
    await page.reload();
    await expect(rows).toHaveCount(before);
  });
});
