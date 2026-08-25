import { expect, type Locator, type Page } from '@playwright/test';

/**
 * Counting record rows, safely.
 *
 * Several app pages stream behind a `loading.tsx` boundary (see
 * app/app/history/loading.tsx), so `page.goto()` resolves as soon as the
 * skeleton is served — the real list arrives afterwards. Playwright's
 * `locator.count()` is one of the few APIs that does NOT auto-wait: it reports
 * whatever is in the DOM at that instant. Calling it straight after `goto()`
 * therefore returns 0 on a slow machine and the right number on a fast one,
 * which is exactly the flake this caused in CI but never locally.
 *
 * Every count in the suite goes through here, which waits for the list or the
 * empty state to actually render first.
 */

/** The record rows on a history page — each carries its own delete disclosure. */
export function recordRows(page: Page): Locator {
  // Scoped by the delete control so it cannot also match the record-type tab
  // bar, which is itself a list.
  return page.locator('main ul > li').filter({ has: page.getByRole('group') });
}

/** Navigate to a history view and count its rows once it has actually rendered. */
export async function countRecordRows(page: Page, historyUrl: string): Promise<number> {
  await page.goto(historyUrl);
  const rows = recordRows(page);
  const emptyState = page.getByText(/No .+ records yet/);
  await expect(rows.first().or(emptyState)).toBeVisible();
  return rows.count();
}
