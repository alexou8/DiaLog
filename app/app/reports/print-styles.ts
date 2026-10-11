/**
 * Print stylesheet for the report page, rendered as a <style> element by the
 * page itself rather than added to `globals.css`, so it exists only while the
 * report is on screen.
 *
 * Everything is scoped through `[data-report-print]`, a marker the page puts on
 * its own root: the app shell (header, side nav, bottom bar) lives in the
 * layout, which this page cannot add classes to, and `:has()` lets the page
 * reach out and hide it without those rules ever applying to another route.
 */
export const REPORT_PRINT_CSS = `
@page { margin: 14mm; }

@media print {
  body:has([data-report-print]) .dl-skip-link,
  body:has([data-report-print]) header.dl-safe-top,
  body:has([data-report-print]) aside,
  body:has([data-report-print]) nav {
    display: none !important;
  }

  /* The shell reserves room for the fixed bottom bar; paper has no such bar. */
  body:has([data-report-print]) main {
    padding: 0 !important;
  }

  /* Paper is white and printers skip backgrounds, so the dark theme's light
     text would print as pale grey on nothing. Print always uses the light
     palette. These mirror the light tokens at the top of globals.css; they are
     a print palette in their own right, and the borders are a little darker
     than on screen because a pale hairline vanishes on a laser printer. */
  :root[data-theme='dark']:has([data-report-print]) {
    color-scheme: light;
    --color-canvas: oklch(100% 0 0);
    --color-surface: oklch(100% 0 0);
    --color-surface-sunken: oklch(96.4% 0.006 95);
    --color-line: oklch(78% 0.01 95);
    --color-line-strong: oklch(62% 0.01 95);
    --color-ink: oklch(24% 0.015 250);
    --color-ink-muted: oklch(40% 0.015 250);
    --color-brand: oklch(48% 0.09 197);
    --color-brand-strong: oklch(40% 0.09 197);
    --color-brand-soft: oklch(95% 0.02 197);
    --color-brand-ink: oklch(30% 0.07 197);
    --color-positive: oklch(46% 0.1 155);
    --color-positive-soft: oklch(95.5% 0.03 155);
    --color-notice: oklch(52% 0.12 65);
    --color-notice-soft: oklch(96% 0.04 75);
    --color-caution: oklch(50% 0.12 40);
    --color-caution-soft: oklch(96% 0.03 40);
    --color-critical: oklch(46% 0.16 20);
    --color-critical-soft: oklch(96% 0.03 20);
    --color-info: oklch(45% 0.1 260);
    --color-info-soft: oklch(96% 0.025 260);
    --dl-on-brand: oklch(100% 0 0);
    --dl-on-critical: oklch(100% 0 0);
  }
}
`;
