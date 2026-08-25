import { chromium } from '@playwright/test';
import path from 'node:path';

const browser = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome' });
const context = await browser.newContext({ storageState: 'tests/e2e/.auth/import.json' });
const page = await context.newPage();
page.on('console', (msg) => console.log('CONSOLE:', msg.type(), msg.text()));
page.on('pageerror', (err) => console.log('PAGEERROR:', err.message));
await page.goto('http://localhost:3100/app/import');
await page.getByLabel(/Choose a file/).setInputFiles(path.join(process.cwd(), 'ml/data/sample_logs.csv'));
await page.getByRole('button', { name: 'Check this file' }).click();
await page.waitForTimeout(1500);
console.log('--- after analyze ---');
console.log((await page.locator('body').innerText()).slice(0, 1500));

const btn = page.getByRole('button', { name: /^Import \d+ record/ });
console.log('button visible?', await btn.isVisible().catch(() => false));
await btn.click();
await page.waitForTimeout(3000);
console.log('--- after commit click ---');
console.log((await page.locator('body').innerText()).slice(0, 1500));
await browser.close();
