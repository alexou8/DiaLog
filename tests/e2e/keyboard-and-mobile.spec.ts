import { uniqueGlucoseValue } from './unique';
import { test, expect, type Page } from '@playwright/test';
import { SHARED_STATE } from './setup/auth-state';

/**
 * This file always runs under the `mobile-chromium` Playwright project (see
 * playwright.config.ts), but each test group sets its own explicit viewport
 * so the "keyboard journey" runs at a normal desktop size while the "mobile"
 * checks run at genuine phone sizes.
 */

async function tabToAccessibleName(page: Page, name: string, maxPresses = 40): Promise<void> {
  for (let i = 0; i < maxPresses; i++) {
    await page.keyboard.press('Tab');
    const info = await page.evaluate(() => {
      const el = document.activeElement as HTMLElement | null;
      return { text: el?.textContent?.trim() ?? '', tag: el?.tagName ?? '' };
    });
    if (
      info.text === name ||
      (info.text.length > 0 && info.text.includes(name) && info.tag === 'A')
    ) {
      return;
    }
  }
  throw new Error(
    `Could not reach an element named "${name}" by tabbing within ${maxPresses} presses.`,
  );
}

test.describe('keyboard-only journey', () => {
  test.use({ viewport: { width: 1280, height: 900 }, storageState: SHARED_STATE });

  test('tab to Add, submit the glucose form with only the keyboard, and see a visible focus indicator', async ({
    page,
  }) => {
    await page.goto('/app');

    // A visible focus indicator exists: tab once and check the computed outline.
    await page.keyboard.press('Tab');
    const outline = await page.evaluate(() => {
      const el = document.activeElement;
      if (!el) return null;
      const style = window.getComputedStyle(el);
      return { outlineStyle: style.outlineStyle, outlineWidth: style.outlineWidth };
    });
    expect(outline).not.toBeNull();
    expect(outline!.outlineStyle).not.toBe('none');
    expect(parseFloat(outline!.outlineWidth)).toBeGreaterThan(0);

    // Continue tabbing from the top until the "Add" action is reached, then
    // activate it. "Add" opens the log panel (a modal dialog listing every
    // kind of entry) rather than going straight to one form, so the journey
    // continues inside the dialog: focus is trapped there, and Tab cycles
    // through its links until "Glucose reading" is reached.
    await page.goto('/app');
    await tabToAccessibleName(page, 'Add');
    await page.keyboard.press('Enter');
    await expect(page.getByRole('dialog', { name: 'What would you like to log?' })).toBeVisible();
    await tabToAccessibleName(page, 'Glucose reading');
    await page.keyboard.press('Enter');

    await expect(page).toHaveURL(/\/app\/glucose\/new/);

    // Fill and submit the glucose form using only the keyboard.
    const valueInput = page.getByLabel(/Your reading/);
    await expect(valueInput).toBeFocused(); // autoFocus on the value field
    // Unique per attempt: dedupe keys are (type, minute, value), so a fixed
    // value re-run inside the same minute is rejected as a duplicate.
    const value = uniqueGlucoseValue(
      (await page.getByText(/^Your reading \(/).textContent()) ?? '',
    );
    await page.keyboard.type(value);
    // Keep tabbing to the submit button rather than counting a fixed number of
    // presses: a radio group's default-checked option (not necessarily the
    // first one) is where Tab lands, so the exact stop count isn't fixed.
    await tabToAccessibleName(page, 'Save reading');
    const active = await page.evaluate(() => document.activeElement?.textContent?.trim());
    expect(active).toBe('Save reading');
    await page.keyboard.press('Enter');

    await expect(page).toHaveURL(/\/app\/glucose\?added=1/);
    // Check the specific reading row rather than any "133" substring on the
    // page: this account is shared with logging.spec.ts, whose concurrent
    // writes can shift aggregate stats (average, median) to other numbers
    // that may or may not contain "133", which would make a page-wide text
    // search flaky. The individual record row is unambiguous regardless.
    await expect(page.locator('li').filter({ hasText: value }).first()).toBeVisible();
  });
});

/** The computed focus indicator of whatever currently has focus. */
async function focusIndicator(page: Page) {
  return page.evaluate(() => {
    const el = document.activeElement as HTMLElement | null;
    if (!el) return null;
    const style = window.getComputedStyle(el);
    return {
      tag: el.tagName,
      outlineStyle: style.outlineStyle,
      outlineWidth: parseFloat(style.outlineWidth),
      boxShadow: style.boxShadow,
    };
  });
}

/** Visible means a real outline or a ring drawn as a box-shadow. */
function expectVisibleFocus(
  indicator: Awaited<ReturnType<typeof focusIndicator>>,
  what: string,
): void {
  expect(indicator, `${what}: something should have focus`).not.toBeNull();
  const outline = indicator!.outlineStyle !== 'none' && indicator!.outlineWidth > 0;
  const ring = indicator!.boxShadow !== 'none' && indicator!.boxShadow !== '';
  expect(outline || ring, `${what}: focus must be visible (got ${JSON.stringify(indicator)})`).toBe(
    true,
  );
}

test.describe('focus is visible on form controls, in both themes', () => {
  test.use({ viewport: { width: 1280, height: 900 }, storageState: SHARED_STATE });

  for (const theme of ['light', 'dark'] as const) {
    test(`an Input and a Button show a focus indicator (${theme} theme)`, async ({ page }) => {
      // The theme is a device-level choice in localStorage, applied before
      // paint by PreferencesScript; set it before the page loads.
      await page.addInitScript((value) => {
        window.localStorage.setItem('dialog-theme', value);
      }, theme);
      await page.goto('/app/glucose/new');
      await expect(page.locator('html')).toHaveAttribute('data-theme', theme);

      // Regression: `outline-none` on the primitives used to beat the global
      // :focus-visible rule through the cascade-layer order, so neither of
      // these had any focus indicator at all.
      const reading = page.getByLabel(/Your reading/);
      await expect(reading).toBeFocused();
      expectVisibleFocus(await focusIndicator(page), 'reading input');

      await tabToAccessibleName(page, 'Save reading');
      const button = await focusIndicator(page);
      expect(button?.tag).toBe('BUTTON');
      expectVisibleFocus(button, 'Save reading button');
    });
  }
});

test.describe('a failed submit keeps what was typed and explains it', () => {
  test.use({ viewport: { width: 1280, height: 900 }, storageState: SHARED_STATE });

  test('error summary takes focus, links to the field, and nothing is wiped', async ({ page }) => {
    await page.goto('/app/glucose/new');
    const note = 'keep this note';
    await page.getByLabel(/A note/).fill(note);
    await page.getByRole('radio', { name: 'At bedtime' }).check();
    // No reading: refused server-side as out of range, so nothing is saved
    // and the shared account's record counts are untouched.
    await page.getByRole('button', { name: 'Save reading' }).click();

    // Scoped by text: Next renders its own empty role="alert" route announcer.
    const summary = page.getByRole('alert').filter({ hasText: 'Please check one answer' });
    await expect(summary).toBeVisible();
    await expect(summary).toBeFocused();
    await expect(page).toHaveURL(/\/app\/glucose\/new/);

    // React 19 resets uncontrolled fields after a form action; the form must
    // not have lost the note or the chosen context.
    await expect(page.getByLabel(/A note/)).toHaveValue(note);
    await expect(page.getByRole('radio', { name: 'At bedtime' })).toBeChecked();

    const reading = page.getByLabel(/Your reading/);
    await expect(reading).toHaveAttribute('aria-invalid', 'true');
    await summary.getByRole('link', { name: /Your reading/ }).click();
    await expect(reading).toBeFocused();
  });
});

const MOBILE_VIEWPORTS = [
  { name: '390x844', width: 390, height: 844 },
  { name: '360x740', width: 360, height: 740 },
];

const MOBILE_PAGES = ['/app', '/app/glucose', '/app/insights', '/app/settings'];

test.describe('mobile layout', () => {
  for (const viewport of MOBILE_VIEWPORTS) {
    test.describe(`viewport ${viewport.name}`, () => {
      test.use({
        viewport: { width: viewport.width, height: viewport.height },
        storageState: SHARED_STATE,
      });

      test(`bottom nav visible, sidebar hidden, no horizontal overflow, 44px targets (${viewport.name})`, async ({
        page,
      }) => {
        // Belt-and-braces: pin the viewport explicitly in addition to the
        // project/describe-level `viewport` option, so a slow/loaded run
        // can't observe a stale size from a not-yet-applied context option.
        await page.setViewportSize({ width: viewport.width, height: viewport.height });

        for (const url of MOBILE_PAGES) {
          await page.goto(url);
          await page.waitForLoadState('networkidle');

          const bottomNav = page.locator('nav[aria-label="Main navigation"]').last();
          const sideNav = page.locator('nav[aria-label="Main navigation"]').first();
          await expect(bottomNav).toBeVisible();
          await expect(sideNav).toBeHidden();

          const overflow = await page.evaluate(() => ({
            scrollWidth: document.documentElement.scrollWidth,
            innerWidth: window.innerWidth,
          }));
          expect(
            overflow.scrollWidth,
            `${url} at ${viewport.name}: horizontal overflow (scrollWidth ${overflow.scrollWidth} > innerWidth ${overflow.innerWidth})`,
          ).toBeLessThanOrEqual(overflow.innerWidth);

          const primaryButton = page
            .getByRole('button', { name: /save|add|ask|check this file/i })
            .first();
          const addLink = page.getByRole('link', { name: 'Add' }).first();
          const target = (await addLink.isVisible().catch(() => false)) ? addLink : primaryButton;
          if (await target.isVisible().catch(() => false)) {
            const box = await target.boundingBox();
            expect(
              box,
              `${url} at ${viewport.name}: primary action should have a bounding box`,
            ).not.toBeNull();
            expect(
              box!.height,
              `${url} at ${viewport.name}: primary action should be at least 44px tall`,
            ).toBeGreaterThanOrEqual(44);
          }
        }
      });
    });
  }
});

/**
 * Every section must be reachable on a phone. Below 1024px the sidebar is
 * hidden, and Meals, Activity, Health, Reports, History and Import used to
 * have no route in at all. The bar is now Home · Glucose · Log · Insights ·
 * More, and both panels are real modal dialogs.
 */
const MORE_DESTINATIONS = [
  { name: 'Meals', href: '/app/meals' },
  { name: 'Activity', href: '/app/activity' },
  { name: 'Health', href: '/app/health' },
  { name: 'Assistant', href: '/app/assistant' },
  { name: 'Reports', href: '/app/reports' },
  { name: 'History', href: '/app/history' },
  { name: 'Import', href: '/app/import' },
  { name: 'Settings', href: '/app/settings' },
];

const LOG_DESTINATIONS = [
  '/app/glucose/new',
  '/app/meals/new',
  '/app/activity/new',
  '/app/health/weight/new',
  '/app/health/blood-pressure/new',
  '/app/health/sleep/new',
  '/app/health/mood/new',
  '/app/health/symptom/new',
  '/app/health/medication/new',
  '/app/health/hydration/new',
  '/app/quick-log',
];

test.describe('mobile navigation reaches every section', () => {
  test.use({ viewport: { width: 390, height: 844 }, storageState: SHARED_STATE });

  test('More lists every other destination, and Escape returns focus', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto('/app');
    const bar = page.locator('nav[aria-label="Main navigation"]').last();
    await expect(bar).toBeVisible();

    const more = bar.getByRole('button', { name: 'More' });
    await more.click();
    const sheet = page.getByRole('dialog', { name: 'More of DiaLog' });
    await expect(sheet).toBeVisible();
    for (const destination of MORE_DESTINATIONS) {
      await expect(sheet.getByRole('link', { name: destination.name })).toHaveAttribute(
        'href',
        destination.href,
      );
    }

    await page.keyboard.press('Escape');
    await expect(sheet).toBeHidden();
    await expect(more).toBeFocused();

    // Following a link closes the panel and lands on the page.
    await more.click();
    await sheet.getByRole('link', { name: 'History' }).click();
    await expect(page).toHaveURL(/\/app\/history$/);
    await expect(page.getByRole('dialog')).toBeHidden();
    // ...with focus on the new page, not back on the "More" button the
    // keyboard user just left behind.
    await expect(page.locator('#main')).toBeFocused();
  });

  test('Log lists every kind of entry', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto('/app');
    const bar = page.locator('nav[aria-label="Main navigation"]').last();
    await bar.getByRole('button', { name: 'Log' }).click();
    const sheet = page.getByRole('dialog', { name: 'What would you like to log?' });
    await expect(sheet).toBeVisible();
    for (const href of LOG_DESTINATIONS) {
      await expect(sheet.locator(`a[href="${href}"]`)).toBeVisible();
    }
    await sheet.getByRole('link', { name: 'Meal', exact: true }).click();
    await expect(page).toHaveURL(/\/app\/meals\/new$/);
  });
});
