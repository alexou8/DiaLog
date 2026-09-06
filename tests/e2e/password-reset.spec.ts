import { createServer, type Server } from 'node:http';
import { randomUUID } from 'node:crypto';
import { PrismaClient } from '@prisma/client';
import AxeBuilder from '@axe-core/playwright';
import { expect, test } from '@playwright/test';
import { provisionSecurityAccount } from './setup/security-accounts';

// The app runs in a separate production-mode process. Its HTTP mail provider
// posts into this loopback relay; the test reads process memory directly, never
// stdout, a mailbox HTTP endpoint, a token database column, or a persisted file.
type CapturedMail = { to: string; text: string };
const messages: CapturedMail[] = [];
const deliveries = new Map<string, (mail: CapturedMail) => void>();
const prisma = new PrismaClient();
let relay: Server;
let email: string;
let statePath: string;

test.use({ trace: 'off', screenshot: 'off' });

test.beforeAll(async () => {
  relay = createServer(async (request, response) => {
    if (
      request.method !== 'POST' ||
      request.headers.authorization !== 'Bearer e2e-mail-relay-key'
    ) {
      response.writeHead(403).end();
      return;
    }
    const chunks: Buffer[] = [];
    for await (const chunk of request) chunks.push(Buffer.from(chunk));
    const message: CapturedMail = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    messages.push(message);
    deliveries.get(message.to)?.(message);
    deliveries.delete(message.to);
    response.writeHead(202).end();
  });
  await new Promise<void>((resolve, reject) => {
    relay.once('error', reject);
    relay.listen(3211, '127.0.0.1', resolve);
  });
  ({ email, statePath } = await provisionSecurityAccount(
    prisma,
    `reset-${randomUUID()}`,
    'http://localhost:3100',
  ));
});
test.afterAll(async () => {
  if (email) await prisma.user.deleteMany({ where: { email } });
  await prisma.$disconnect();
  if (relay)
    await new Promise<void>((resolve, reject) =>
      relay.close((error) => (error ? reject(error) : resolve())),
    );
});

test('request mail, reset, revoke an existing session, and sign in with the new password', async ({
  page,
  browser,
}) => {
  const oldContext = await browser.newContext({ storageState: statePath });
  const oldPage = await oldContext.newPage();
  try {
    await oldPage.goto('/app/settings');
    await expect(oldPage.getByRole('heading', { name: 'Change your password' })).toBeVisible();
    // An unverifiable cookie must not keep somebody off either recovery page.
    await page
      .context()
      .addCookies([
        { name: 'dialog_session', value: 'unverifiable', domain: 'localhost', path: '/' },
      ]);
    await page.goto('/sign-in');
    await page.getByRole('link', { name: 'Forgot your password?' }).click();
    // Wait for the destination before touching the field. /sign-in has its own
    // "Email address" input, so filling straight after the click raced the
    // client-side navigation: the value landed in the sign-in form, the
    // forgot-password form then rendered empty, and the action received a
    // blank email. Asserting URL and heading polls until the swap is done, so
    // it stays correct on a slow runner where a fixed delay would not.
    await expect(page).toHaveURL(/\/forgot-password$/);
    await expect(page.getByRole('heading', { name: 'Forgot your password?' })).toBeVisible();
    await page.getByLabel('Email address').fill(email);
    // Delivery now starts after the response. Subscribe before submitting and
    // await the actual relay event, never race a count assertion or add a sleep.
    const delivered = new Promise<CapturedMail>((resolve) => deliveries.set(email, resolve));
    await page.getByRole('button', { name: 'Send reset link' }).click();
    await expect(
      page.getByText('If an account with a password matches that email', { exact: false }),
    ).toBeVisible();
    const message = await delivered;
    expect(messages.filter((m) => m.to === email)).toHaveLength(1);
    const url = message.text.split('\n').find((line) => line.startsWith('http'))!;
    const raw = new URL(url).searchParams.get('token')!;
    const response = await page.goto(url);
    await expect(
      page.getByRole('heading', { name: 'Reset your password', exact: true }),
    ).toBeVisible();
    expect((await response!.text()).includes(raw)).toBe(false);
    expect(page.url().includes(raw)).toBe(false);
    expect(
      (
        await new AxeBuilder({
          // Same adapter as accessibility.spec.ts: axe's nested Playwright types
          // differ from the runner's pinned types; the runtime Page API matches.
          page: page as unknown as ConstructorParameters<typeof AxeBuilder>[0]['page'],
        })
          .withTags(['wcag2a', 'wcag2aa', 'wcag21aa', 'wcag22aa'])
          .analyze()
      ).violations,
    ).toEqual([]);

    await page.getByLabel('New password', { exact: false }).first().fill('short');
    await page.getByLabel('Confirm new password').fill('short');
    await page.getByRole('button', { name: 'Reset password', exact: true }).click();
    await expect(page.locator('input[name="newPassword"]')).toHaveAttribute('aria-invalid', 'true');
    const password = 'a-new-e2e-recovery-password-2026';
    await page.locator('input[name="newPassword"]').fill(password);
    await page.getByLabel('Confirm new password').fill(password);
    await page.getByRole('button', { name: 'Reset password', exact: true }).click();
    await expect(page).toHaveURL(/\/sign-in\?password=reset$/);
    await expect(
      page.getByText('Your password has been reset. Sign in with your new password.'),
    ).toBeVisible();
    await oldPage.goto('/app/settings');
    await expect(oldPage).toHaveURL(/\/sign-in/);
    await page.getByLabel('Email address').fill(email);
    await page.getByLabel('Password', { exact: false }).fill(password);
    await page.getByRole('button', { name: 'Sign in', exact: true }).click();
    await expect(page).toHaveURL(/\/app(?:\/|$)/);
    await page.goto(url);
    // Scoped to the Callout's text: a bare getByRole('alert') also matches
    // Next's always-present __next-route-announcer__ live region, which makes
    // the locator ambiguous under Playwright strict mode.
    await expect(
      page.getByRole('alert').filter({ hasText: 'invalid or has expired' }),
    ).toBeVisible();
  } finally {
    await oldContext.close();
  }
});

test('unknown email receives the generic confirmation and no mail', async ({ page }) => {
  const unknown = `unknown-${randomUUID()}@dialog.test`;
  const requestedAt = new Date();
  await page.goto('/forgot-password');
  await page.getByLabel('Email address').fill(unknown);
  await page.getByRole('button', { name: 'Send reset link' }).click();
  await expect(
    page.getByText('If an account with a password matches that email', { exact: false }),
  ).toBeVisible();
  // Wait for the unknown-account callback to reach its audit before asserting
  // no mail: the neutral response intentionally arrives before this work now.
  await expect
    .poll(() =>
      prisma.auditEvent.findFirst({
        where: {
          action: 'auth.password_reset_requested',
          userId: null,
          createdAt: { gte: requestedAt },
        },
        select: { id: true },
      }),
    )
    .not.toBeNull();
  expect(messages.some((m) => m.to === unknown)).toBe(false);
  expect(
    (
      await new AxeBuilder({
        // Same adapter as accessibility.spec.ts: axe's nested Playwright types
        // differ from the runner's pinned types; the runtime Page API matches.
        page: page as unknown as ConstructorParameters<typeof AxeBuilder>[0]['page'],
      })
        .withTags(['wcag2a', 'wcag2aa', 'wcag21aa', 'wcag22aa'])
        .analyze()
    ).violations,
  ).toEqual([]);
});
