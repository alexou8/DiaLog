import { chromium } from '@playwright/test';

const browser = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome' });
const context = await browser.newContext({ storageState: 'tests/e2e/.auth/demo.json' });
const page = await context.newPage();
await page.goto('http://localhost:3100/app');
const info = await page.evaluate(() => {
  const el = document.querySelector('a[href="/app/glucose/new"]');
  if (!el) return null;
  const rect = el.getBoundingClientRect();
  const cs = window.getComputedStyle(el);
  return {
    rect: { width: rect.width, height: rect.height },
    display: cs.display,
    minHeight: cs.minHeight,
    padding: cs.padding,
    lineHeight: cs.lineHeight,
    fontSize: cs.fontSize,
    className: el.className,
    outerHTML: el.outerHTML.slice(0, 300),
  };
});
console.log(JSON.stringify(info, null, 2));
await browser.close();
