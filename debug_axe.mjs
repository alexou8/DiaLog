import { chromium } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

const browser = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome' });
const context = await browser.newContext({ storageState: 'tests/e2e/.auth/demo.json' });
const page = await context.newPage();
await page.goto('http://localhost:3100/app');
const results = await new AxeBuilder({ page }).withTags(['wcag22aa']).analyze();
const targetSize = results.violations.find(v => v.id === 'target-size');
console.log(JSON.stringify(targetSize?.nodes?.slice(0,2), null, 2));
await browser.close();
