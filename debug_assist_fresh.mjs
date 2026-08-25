import { chromium } from '@playwright/test';

const browser = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome' });
const context = await browser.newContext({ storageState: 'tests/e2e/.auth/assistant-fresh.json' });
const page = await context.newPage();
const start = Date.now();
await page.goto('http://localhost:3100/app/assistant');
console.log('goto took', Date.now() - start, 'ms');
await page.waitForLoadState('networkidle');
console.log('networkidle at', Date.now() - start, 'ms');
const t0 = Date.now();
await page.getByLabel('Your question').fill('Why were my readings higher this week?');
console.log('fill took', Date.now() - t0, 'ms');
await page.getByRole('button', { name: 'Ask', exact: true }).click();
await page.waitForSelector('text=You asked', { timeout: 20000 });
console.log('answered at', Date.now() - start, 'ms');
await browser.close();
