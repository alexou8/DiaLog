/**
 * A guardrail, not a behaviour test.
 *
 * Revoking sessions means bumping `User.tokenVersion`, which invalidates every
 * outstanding cookie for the account — including the one carried by the request
 * doing the bumping. A Server Action cannot report that outcome reliably: the
 * client router has other requests for the same document in flight, all of them
 * still holding the revoked cookie, and any that is served after the bump comes
 * back as a redirect to /sign-in, which makes the router discard the action's
 * result. That is the "6 failures in 10" change-password defect.
 *
 * The fix was to move every revoking mutation onto a route handler posted to by
 * a plain HTML form, so the browser owns the navigation and no request holding
 * the old cookie can outlive it (lib/auth/route-form.ts).
 *
 * Nothing about a Server Action stops someone reintroducing the same shape a
 * year from now — the code would look entirely reasonable and fail one time in
 * two, only under a real browser. So this test fails the build instead.
 */
import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';

const ACTIONS_DIR = join(process.cwd(), 'lib/actions');

/**
 * A *write* to tokenVersion, not a read.
 *
 * `select: { tokenVersion: true }` and `tokenVersion: user.tokenVersion` (which
 * is how sign-in mints a token) are both fine and must keep passing. What is
 * forbidden is putting a value into the column: `{ increment: 1 }`, or a
 * literal.
 */
const WRITE_PATTERNS = [/tokenVersion\s*:\s*\{/, /tokenVersion\s*:\s*-?\d/];

function actionModules(): string[] {
  return readdirSync(ACTIONS_DIR).filter((name) => name.endsWith('.ts'));
}

describe('session revocation stays out of Server Actions', () => {
  it('finds the action modules it is supposed to be checking', () => {
    // Guards the guard: a rename that empties this list must not read as a pass.
    expect(actionModules().length).toBeGreaterThan(0);
  });

  it.each(actionModules())('%s does not write User.tokenVersion', (name) => {
    const source = readFileSync(join(ACTIONS_DIR, name), 'utf8');
    const offending = WRITE_PATTERNS.filter((pattern) => pattern.test(source));

    expect(
      offending,
      `lib/actions/${name} writes User.tokenVersion.\n\n` +
        'Bumping tokenVersion revokes the cookie the current request arrived with, ' +
        'and a Server Action cannot deliver that result through the client router — ' +
        'it fails intermittently, in a browser only, with the mutation already committed.\n\n' +
        'Put the mutation in a route handler posted to by a plain <form method="post">, ' +
        'the way app/api/auth/password/route.ts and ' +
        'app/api/auth/sessions/revoke/route.ts do. See lib/auth/route-form.ts.',
    ).toEqual([]);
  });
});

describe('the revoking route handlers keep their shape', () => {
  const HANDLERS = [
    'app/api/auth/password/route.ts',
    'app/api/auth/sessions/revoke/route.ts',
    'app/api/auth/google/disconnect/route.ts',
  ];

  it.each(HANDLERS)('%s answers with a redirect and checks its own origin', (path) => {
    const source = readFileSync(join(process.cwd(), path), 'utf8');
    // Route handlers get no CSRF protection from Next, so each must do its own.
    expect(source, `${path} must reject cross-origin posts`).toContain('isSameOrigin');
    // 303 keeps the browser's follow-up a GET, so a reload cannot re-submit.
    expect(source, `${path} must reply via the shared 303 helpers`).toMatch(/back\(|toSignIn\(/);
  });
});

describe('the settings forms that revoke sessions post natively', () => {
  const SOURCE = readFileSync(join(process.cwd(), 'app/app/settings/AccountForms.tsx'), 'utf8');

  it.each([
    ['/api/auth/password', 'ChangePasswordForm'],
    ['/api/auth/sessions/revoke', 'SignOutEverywhereForm'],
  ])('%s is submitted by a native form', (endpoint) => {
    expect(SOURCE).toContain(`action="${endpoint}"`);
  });

  it('does not bind either form to a Server Action', () => {
    // `action={...}` on these two would put the response back through the
    // client router, which is the whole defect.
    const forms = SOURCE.split(
      '// -------------------------------------------------------- destructive zone',
    )[0];
    expect(forms).not.toMatch(/changePasswordAction|signOutEverywhereAction/);
  });
});
