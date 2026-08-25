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

    // Continue tabbing from the top until the "Add" action is reached, then activate it.
    await page.goto('/app');
    await tabToAccessibleName(page, 'Add');
    await page.keyboard.press('Enter');

    await expect(page).toHaveURL(/\/app\/glucose\/new/);

    // Fill and submit the glucose form using only the keyboard.
    const valueInput = page.getByLabel(/Your reading/);
    await expect(valueInput).toBeFocused(); // autoFocus on the value field
    await page.keyboard.type('133');
    await page.keyboard.press('Tab'); // -> When was it taken (leave default)
    await page.keyboard.press('Tab'); // -> first context radio (Fasting)
    await page.keyboard.press('ArrowRight'); // -> Before a meal
    await page.keyboard.press('Tab'); // -> note field
    await page.keyboard.press('Tab'); // -> submit button
    const active = await page.evaluate(() => document.activeElement?.textContent?.trim());
    expect(active).toBe('Save reading');
    await page.keyboard.press('Enter');

    await expect(page).toHaveURL(/\/app\/glucose\?added=1/);
    await expect(page.getByText('133').first()).toBeVisible();
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
        for (const url of MOBILE_PAGES) {
          await page.goto(url);

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
