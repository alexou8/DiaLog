import { chromium } from '@playwright/test';
import path from 'node:path';

const browser = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome' });
const context = await browser.newContext({ storageState: 'tests/e2e/.auth/import.json' });
const page = await context.newPage();
page.on('framenavigated', (f) => console.log('NAVIGATED:', f.url()));
page.on('request', (r) => { if (r.method() === 'POST') console.log('POST:', r.url()); });
await page.goto('http://localhost:3100/app/import');
await page.getByLabel(/Choose a file/).setInputFiles(path.join(process.cwd(), 'ml/data/sample_logs.csv'));

const before = await page.evaluate(() => document.getElementById('import-file')?.files?.length);
console.log('files before submit:', before);

await page.getByRole('button', { name: 'Check this file' }).click();
await page.waitForTimeout(1500);

const after = await page.evaluate(() => document.getElementById('import-file')?.files?.length);
console.log('files after analyze submit:', after);
await browser.close();
