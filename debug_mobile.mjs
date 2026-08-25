import { chromium, devices } from '@playwright/test';

const browser = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome' });
const context = await browser.newContext({ storageState: 'tests/e2e/.auth/shared.json', viewport: { width: 390, height: 844 } });
const page = await context.newPage();
await page.goto('http://localhost:3100/app');
const navs = await page.evaluate(() => {
  return Array.from(document.querySelectorAll('nav[aria-label="Main navigation"]')).map((n) => {
    const cs = window.getComputedStyle(n);
    const rect = n.getBoundingClientRect();
    return { className: n.className, display: cs.display, rect: { w: rect.width, h: rect.height } };
  });
});
console.log(JSON.stringify(navs, null, 2));
console.log('viewport width used by page:', await page.evaluate(() => window.innerWidth));
await browser.close();
