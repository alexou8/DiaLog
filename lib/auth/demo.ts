/**
 * The shared, publicly documented demo account.
 *
 * prisma/seed.ts creates it with a password that is printed in the README, so
 * anyone can sign in as it. That is the point of a demo, but it also means any
 * visitor could change its password, sign everyone else out, attach their own
 * Google account as a backdoor, or delete it — locking every later visitor out
 * of the demo until the next re-seed. The account is therefore read-only for
 * credentials and account-wide deletion, and it can never grant consent to
 * external AI (a shared consent would cover every later visitor's free text).
 * Deleting single records and undoing an import stay allowed on purpose: they
 * are part of trying the app, and the scheduled re-seed restores the data.
 *
 * Matched on email because there is no `isDemo` column (adding one is a schema
 * change outside this module). Keep DEMO_EMAIL in step with prisma/seed.ts —
 * tests/unit/auth/demo.test.ts fails if the two drift apart.
 */
export const DEMO_EMAIL = 'demo@dialog.health';

/**
 * Shown when the demo account tries something it is not allowed to. It says
 * what happened and what to do instead, and it names nothing about any other
 * account — the person already knows they are signed in to the demo.
 */
export const DEMO_READ_ONLY_MESSAGE =
  'This is the shared demo account, so its password, sign-in methods, external AI sharing and deletion are locked for everyone. Create your own free account to try these.';

export function isDemoEmail(email: string | null | undefined): boolean {
  return !!email && email.trim().toLowerCase() === DEMO_EMAIL;
}
