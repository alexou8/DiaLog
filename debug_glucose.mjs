import { chromium } from '@playwright/test';

const browser = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome' });
const context = await browser.newContext({ storageState: 'tests/e2e/.auth/shared.json' });
const page = await context.newPage();
page.on('console', (msg) => console.log('CONSOLE:', msg.type(), msg.text()));
page.on('pageerror', (err) => console.log('PAGEERROR:', err.message));
await page.goto('http://localhost:3100/app/glucose/new');
await page.getByLabel(/Your reading/).fill('142');
await page.getByRole('radio', { name: 'After a meal', exact: false }).check();
await page.getByRole('button', { name: 'Save reading' }).click();
await page.waitForTimeout(2000);
console.log('URL after click:', page.url());
const bodyText = await page.locator('body').innerText();
console.log(bodyText.slice(0, 800));
await browser.close();
