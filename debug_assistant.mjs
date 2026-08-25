import { chromium } from '@playwright/test';

const browser = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome' });
const context = await browser.newContext({ storageState: 'tests/e2e/.auth/assistant-fresh.json' });
const page = await context.newPage();
await page.goto('http://localhost:3100/app/assistant');
console.log('URL:', page.url());
console.log((await page.locator('body').innerText()).slice(0, 500));
await browser.close();
