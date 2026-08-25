import { chromium } from '@playwright/test';

const browser = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome' });
const context = await browser.newContext({ storageState: 'tests/e2e/.auth/assistant-fresh.json' });
const page = await context.newPage();
page.on('pageerror', (e) => console.log('PAGEERROR:', e.message));
page.on('console', (m) => console.log('CONSOLE', m.type(), m.text()));
await page.goto('http://localhost:3100/app/assistant');
await page.waitForLoadState('networkidle');
console.log('URL:', page.url());
console.log((await page.locator('body').innerText()).slice(0, 1500));
const labelCount = await page.locator('label').count();
console.log('label count:', labelCount);
const labels = await page.locator('label').allTextContents();
console.log('labels:', labels);
await browser.close();
