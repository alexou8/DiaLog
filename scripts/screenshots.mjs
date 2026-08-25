/** Captures the screenshots used in the README. Requires the app on :3000. */
import { chromium, devices } from 'playwright';
import { mkdirSync } from 'node:fs';

const BASE = process.env.SHOT_BASE ?? 'http://localhost:3000';
const OUT = 'docs/screenshots';
mkdirSync(OUT, { recursive: true });

const DEMO = { email: 'demo@dialog.health', password: 'demo-account-2026' };

async function signIn(page) {
  await page.goto(`${BASE}/sign-in`, { waitUntil: 'networkidle' });
  await page.getByLabel('Email address').fill(DEMO.email);
  await page.getByLabel('Password').fill(DEMO.password);
  await page.getByRole('button', { name: 'Sign in' }).click();
  await page.waitForURL(`${BASE}/app**`, { timeout: 30000 });
}

async function shot(page, path, name, { full = true } = {}) {
  await page.goto(`${BASE}${path}`, { waitUntil: 'networkidle' });
  await page.waitForTimeout(700);
  await page.screenshot({ path: `${OUT}/${name}.png`, fullPage: full });
  console.log('captured', name);
}

const browser = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium' });

// --- Desktop
const desktop = await browser.newContext({ viewport: { width: 1280, height: 900 }, deviceScaleFactor: 2 });
const page = await desktop.newPage();

await shot(page, '/', 'landing');
await signIn(page);
await shot(page, '/app', 'dashboard');
await shot(page, '/app/glucose', 'glucose');
await shot(page, '/app/insights', 'insights');
await shot(page, '/app/import', 'import');
await shot(page, '/app/reports', 'reports');
await shot(page, '/app/settings', 'settings');
await shot(page, '/app/glucose/new', 'add-reading');
await shot(page, '/app/history', 'history');

// Assistant: ask a real question so the screenshot shows a real answer.
await page.goto(`${BASE}/app/assistant`, { waitUntil: 'networkidle' });
await page.getByLabel('Your question').fill('Does walking after dinner seem to help me?');
await page.getByRole('button', { name: 'Ask', exact: true }).click();
await page.waitForTimeout(4000);
await page.screenshot({ path: `${OUT}/assistant.png`, fullPage: true });
console.log('captured assistant');

// --- Dark theme
await page.emulateMedia({ colorScheme: 'dark' });
await page.goto(`${BASE}/app`, { waitUntil: 'networkidle' });
await page.evaluate(() => localStorage.setItem('dialog-theme', 'dark'));
await page.reload({ waitUntil: 'networkidle' });
await page.waitForTimeout(700);
await page.screenshot({ path: `${OUT}/dashboard-dark.png`, fullPage: true });
console.log('captured dashboard-dark');

// --- Mobile
const mobile = await browser.newContext({ ...devices['iPhone 13'], deviceScaleFactor: 3 });
const mpage = await mobile.newPage();
await signIn(mpage);
await shot(mpage, '/app', 'mobile-dashboard');
await shot(mpage, '/app/glucose', 'mobile-glucose');
await shot(mpage, '/app/glucose/new', 'mobile-add-reading');

await browser.close();
console.log('done');
