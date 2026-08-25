import { chromium } from '@playwright/test';

const browser = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome' });
const context = await browser.newContext({ storageState: 'tests/e2e/.auth/shared.json' });
const page = await context.newPage();
await page.goto('http://localhost:3100/app/glucose/new');
const before = await page.evaluate(() => document.activeElement?.tagName + '#' + document.activeElement?.id);
console.log('active before blur:', before);
await page.evaluate(() => (document.activeElement)?.blur());
const afterBlur = await page.evaluate(() => document.activeElement?.tagName);
console.log('active after blur:', afterBlur);
await page.keyboard.press('Tab');
const afterTab = await page.evaluate(() => ({ tag: document.activeElement?.tagName, cls: document.activeElement?.className, text: document.activeElement?.textContent?.trim().slice(0,40) }));
console.log('active after tab:', afterTab);
await browser.close();
