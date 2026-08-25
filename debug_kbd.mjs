import { chromium } from '@playwright/test';

const browser = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome' });
const context = await browser.newContext({ storageState: 'tests/e2e/.auth/shared.json', viewport: { width: 1280, height: 900 } });
const page = await context.newPage();
await page.goto('http://localhost:3100/app/glucose/new');
const valueInput = page.getByLabel(/Your reading/);
await valueInput.focus();
await page.keyboard.type('133');
console.log('input value now:', await valueInput.inputValue());

async function tabToAccessibleName(name, maxPresses = 40) {
  for (let i = 0; i < maxPresses; i++) {
    await page.keyboard.press('Tab');
    const info = await page.evaluate(() => {
      const el = document.activeElement;
      return { text: el?.textContent?.trim() ?? '', tag: el?.tagName ?? '' };
    });
    console.log(i, info);
    if (info.text === name) return true;
  }
  return false;
}
const found = await tabToAccessibleName('Save reading');
console.log('found submit:', found);
console.log('input value before submit:', await valueInput.inputValue());
await page.keyboard.press('Enter');
await page.waitForURL(/added=1/);
console.log('after submit url:', page.url());
const body = await page.locator('body').innerText();
console.log(body.includes('133') ? 'FOUND 133 in body' : 'NOT FOUND 133 in body');
console.log(body.slice(0, 1000));
await browser.close();
