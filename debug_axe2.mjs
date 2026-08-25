import { chromium } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

const browser = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome' });
const context = await browser.newContext({ storageState: 'tests/e2e/.auth/demo.json', viewport: { width: 1280, height: 720 } });
const page = await context.newPage();
await page.goto('http://localhost:3100/app');
const results = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22aa']).analyze();
console.log('violations:', results.violations.map(v => v.id));
const targetSize = results.violations.find(v => v.id === 'target-size');
console.log(JSON.stringify(targetSize?.nodes?.slice(0,1), null, 2));
await browser.close();
