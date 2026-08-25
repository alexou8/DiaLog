import { chromium } from '@playwright/test';
import path from 'node:path';

const browser = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome' });
const context = await browser.newContext({ storageState: 'tests/e2e/.auth/import.json' });
const page = await context.newPage();
await page.goto('http://localhost:3100/app/import');
await page.getByLabel(/Choose a file/).setInputFiles(path.join(process.cwd(), 'ml/data/sample_logs.csv'));
await page.getByRole('button', { name: 'Check this file' }).click();
await page.waitForTimeout(1500);

const filesInfo = await page.evaluate(() => {
  const input = document.getElementById('import-file');
  return { count: input?.files?.length, name: input?.files?.[0]?.name };
});
console.log('file input state before commit click:', filesInfo);
await browser.close();
