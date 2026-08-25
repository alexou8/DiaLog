import { test, expect, type Page } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { DEMO_STATE } from './setup/auth-state';

const SERIOUS_IMPACTS = new Set(['serious', 'critical']);

async function runAxe(page: Page) {
  // @axe-core/playwright's type declarations pin a `playwright-core` Page
  // type that structurally diverges from the one re-exported by
  // `@playwright/test` in this workspace, even though they are the same
  // object at runtime. The cast bridges the two types without using `any`.
  return new AxeBuilder({
    page: page as unknown as ConstructorParameters<typeof AxeBuilder>[0]['page'],
  })
    .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22aa'])
    .analyze();
}

async function assertNoSeriousViolations(page: Page, label: string) {
  const results = await runAxe(page);
  const serious = results.violations.filter((v) => SERIOUS_IMPACTS.has(v.impact ?? ''));
  const minor = results.violations.filter((v) => !SERIOUS_IMPACTS.has(v.impact ?? ''));

  if (minor.length > 0) {
    // Reported, not failed: moderate/minor findings are worth knowing about
    // but should not block the suite.
    console.log(
      `[axe] ${label}: ${minor.length} moderate/minor finding(s): ` +
        minor.map((v) => `${v.id} (${v.impact})`).join(', '),
    );
  }

  expect(
    serious,
    `${label}: serious/critical axe violations found:\n${serious
      .map(
        (v) =>
          `- ${v.id} (${v.impact}): ${v.help}\n  ${v.nodes.map((n) => n.target.join(' ')).join('; ')}`,
      )
      .join('\n')}`,
  ).toEqual([]);
}

/** Exactly one h1, and no heading level is skipped going down the page. */
async function assertHeadingStructure(page: Page, label: string) {
  const levels = await page.evaluate(() =>
    Array.from(document.querySelectorAll('h1, h2, h3, h4, h5, h6')).map((el) =>
      Number(el.tagName[1]),
    ),
  );

  const h1Count = levels.filter((l) => l === 1).length;
  expect(
    h1Count,
    `${label}: expected exactly one <h1>, found ${h1Count} (levels: ${levels.join(',')})`,
  ).toBe(1);

  for (let i = 1; i < levels.length; i++) {
    const prev = levels[i - 1]!;
    const cur = levels[i]!;
    expect(
      cur - prev,
      `${label}: heading level skipped going from h${prev} to h${cur} (full sequence: ${levels.join(',')})`,
    ).toBeLessThanOrEqual(1);
  }
}

async function assertHtmlLang(page: Page, label: string) {
  const lang = await page.evaluate(() => document.documentElement.getAttribute('lang'));
  expect(lang, `${label}: <html lang> should be set`).toBeTruthy();
}

/** The skip link is off-screen until focused, and moves focus into <main>. */
async function assertSkipLink(page: Page, label: string) {
  const skipLink = page.locator('.dl-skip-link');

  // The skip link must be the first focusable element in the document, so a
  // keyboard user starting from the very top of the page reaches it on the
  // first Tab. Checked via DOM order rather than a literal simulated Tab
  // press: some pages (e.g. the glucose entry form) legitimately autoFocus a
  // field on load, and per the HTML focus-navigation spec that field — not
  // the top of the document — becomes the "sequential focus navigation
  // starting point" for Tab from then on, which would make a literal Tab
  // press an unreliable way to test DOM order specifically.
  const firstFocusableIsSkipLink = await page.evaluate(() => {
    const candidates = Array.from(
      document.querySelectorAll<HTMLElement>(
        'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]',
      ),
    ).filter((el) => el.tabIndex !== -1);
    return candidates[0]?.classList.contains('dl-skip-link') ?? false;
  });
  expect(
    firstFocusableIsSkipLink,
    `${label}: the skip link should be the first focusable element in the document`,
  ).toBe(true);

  // Focus it directly (equivalent to what Shift+Tab from anywhere past it, or
  // Tab from a page with nothing pre-focused, would do) and confirm it comes
  // on-screen while focused.
  await skipLink.focus();
  await expect(skipLink, `${label}: the skip link should be focusable`).toBeFocused();

  const box = await skipLink.boundingBox();
  expect(box, `${label}: focused skip link should have a visible position`).not.toBeNull();
  expect(
    box!.x,
    `${label}: focused skip link should be on-screen (not offset off-canvas)`,
  ).toBeGreaterThanOrEqual(0);

  await page.keyboard.press('Enter');
  const isMainFocused = await page.evaluate(() => document.activeElement?.id === 'main');
  const mainContainsFocus = await page.evaluate(() => {
    const main = document.getElementById('main');
    return !!main && !!document.activeElement && main.contains(document.activeElement);
  });
  expect(
    isMainFocused || mainContainsFocus,
    `${label}: activating the skip link should move focus to or inside <main>`,
  ).toBe(true);
}

const PUBLIC_PAGES = ['/', '/privacy', '/accessibility', '/sign-in', '/sign-up'];
const APP_PAGES = [
  '/app',
  '/app/glucose',
  '/app/glucose/new',
  '/app/insights',
  '/app/import',
  '/app/settings',
  '/app/history',
];

test.describe('accessibility: public pages', () => {
  for (const url of PUBLIC_PAGES) {
    test(`axe + structure: ${url}`, async ({ page }) => {
      await page.goto(url);
      // Layout must have settled (fonts/styles applied) before measuring
      // target sizes, otherwise axe's target-size check can sample an interim,
      // unstyled layout under CPU contention and report false positives.
      await page.waitForLoadState('networkidle');
      await assertNoSeriousViolations(page, url);
      await assertHeadingStructure(page, url);
      await assertHtmlLang(page, url);
      await assertSkipLink(page, url);
    });
  }
});

test.describe('accessibility: authenticated app pages', () => {
  test.use({ storageState: DEMO_STATE });

  for (const url of APP_PAGES) {
    test(`axe + structure: ${url}`, async ({ page }) => {
      await page.goto(url);
      // Layout must have settled (fonts/styles applied) before measuring
      // target sizes, otherwise axe's target-size check can sample an interim,
      // unstyled layout under CPU contention and report false positives.
      await page.waitForLoadState('networkidle');
      await assertNoSeriousViolations(page, url);
      await assertHeadingStructure(page, url);
      await assertHtmlLang(page, url);
      await assertSkipLink(page, url);
    });
  }
});
