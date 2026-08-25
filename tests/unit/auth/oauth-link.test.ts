import { describe, expect, it } from 'vitest';
import {
  OAUTH_MESSAGES,
  oauthMessage,
  resolveGoogleLink,
  resolveGoogleSignIn,
  type LinkSnapshot,
  type OAuthErrorCode,
  type SignInSnapshot,
} from '@/lib/auth/oauth/link';
import { safeNext } from '@/lib/auth/oauth/state';
import type { GoogleIdentity } from '@/lib/auth/oauth/google';

function identity(overrides: Partial<GoogleIdentity> = {}): GoogleIdentity {
  return {
    subject: 'google-subject-1',
    email: 'person@example.com',
    emailVerified: true,
    name: 'Person One',
    ...overrides,
  };
}

function signInSnapshot(overrides: Partial<SignInSnapshot> = {}): SignInSnapshot {
  return {
    identityUserId: null,
    userWithEmail: null,
    ...overrides,
  };
}

function linkSnapshot(overrides: Partial<LinkSnapshot> = {}): LinkSnapshot {
  return {
    identityUserId: null,
    currentUserId: 'user-current',
    currentUserHasGoogle: false,
    ...overrides,
  };
}

describe('resolveGoogleSignIn', () => {
  it('blocks an unverified Google email', () => {
    const outcome = resolveGoogleSignIn(identity({ emailVerified: false }), signInSnapshot());
    expect(outcome).toEqual({ kind: 'blocked', code: 'unverified_email' });
  });

  it('blocks unverified email even when the subject is already linked', () => {
    const outcome = resolveGoogleSignIn(
      identity({ emailVerified: false }),
      signInSnapshot({ identityUserId: 'user-1' }),
    );
    expect(outcome).toEqual({ kind: 'blocked', code: 'unverified_email' });
  });

  it('signs in a known subject even when the email now matches a different account', () => {
    const outcome = resolveGoogleSignIn(
      identity({ email: 'renamed@example.com' }),
      signInSnapshot({
        identityUserId: 'user-known',
        userWithEmail: { id: 'user-someone-else', hasPassword: true },
      }),
    );
    expect(outcome).toEqual({ kind: 'sign_in', userId: 'user-known' });
  });

  it('blocks with email_in_use when the email collides with an existing password account', () => {
    const outcome = resolveGoogleSignIn(
      identity(),
      signInSnapshot({ userWithEmail: { id: 'user-2', hasPassword: true } }),
    );
    expect(outcome).toEqual({
      kind: 'blocked',
      code: 'email_in_use',
      email: 'person@example.com',
    });
  });

  it('blocks with email_in_use when the email collides with a passwordless account', () => {
    const outcome = resolveGoogleSignIn(
      identity(),
      signInSnapshot({ userWithEmail: { id: 'user-3', hasPassword: false } }),
    );
    expect(outcome).toEqual({
      kind: 'blocked',
      code: 'email_in_use',
      email: 'person@example.com',
    });
  });

  it('creates a new account for a brand-new email with no existing identity', () => {
    const outcome = resolveGoogleSignIn(identity(), signInSnapshot());
    expect(outcome).toEqual({ kind: 'create' });
  });
});

describe('resolveGoogleLink', () => {
  it('blocks an unverified Google email', () => {
    const outcome = resolveGoogleLink(identity({ emailVerified: false }), linkSnapshot());
    expect(outcome).toEqual({ kind: 'blocked', code: 'unverified_email' });
  });

  it('reports already_linked when the current user already holds this Google identity', () => {
    const outcome = resolveGoogleLink(
      identity(),
      linkSnapshot({ identityUserId: 'user-current', currentUserId: 'user-current' }),
    );
    expect(outcome).toEqual({ kind: 'already_linked' });
  });

  it('blocks with linked_elsewhere when the identity belongs to another user', () => {
    const outcome = resolveGoogleLink(
      identity(),
      linkSnapshot({ identityUserId: 'user-other', currentUserId: 'user-current' }),
    );
    expect(outcome).toEqual({ kind: 'blocked', code: 'linked_elsewhere' });
  });

  it('blocks with provider_already_linked when the current account already has a Google identity', () => {
    const outcome = resolveGoogleLink(
      identity(),
      linkSnapshot({ identityUserId: null, currentUserHasGoogle: true }),
    );
    expect(outcome).toEqual({ kind: 'blocked', code: 'provider_already_linked' });
  });

  it('links when the identity is unclaimed and the account has no Google identity yet', () => {
    const outcome = resolveGoogleLink(identity(), linkSnapshot());
    expect(outcome).toEqual({ kind: 'link' });
  });

  it('prioritizes already_linked over provider_already_linked when both would apply', () => {
    const outcome = resolveGoogleLink(
      identity(),
      linkSnapshot({
        identityUserId: 'user-current',
        currentUserId: 'user-current',
        currentUserHasGoogle: true,
      }),
    );
    expect(outcome).toEqual({ kind: 'already_linked' });
  });
});

describe('oauthMessage', () => {
  it('returns null for null, undefined, and empty input', () => {
    expect(oauthMessage(null)).toBeNull();
    expect(oauthMessage(undefined)).toBeNull();
    expect(oauthMessage('')).toBeNull();
  });

  it('returns null for an unknown code', () => {
    expect(oauthMessage('not_a_real_code')).toBeNull();
  });

  it('returns the matching message for a known code', () => {
    expect(oauthMessage('email_in_use')).toBe(OAUTH_MESSAGES.email_in_use);
  });

  it('has a non-empty message for every OAuthErrorCode', () => {
    const codes: OAuthErrorCode[] = [
      'unverified_email',
      'email_in_use',
      'linked_elsewhere',
      'provider_already_linked',
      'not_configured',
      'invalid_state',
      'access_denied',
      'exchange_failed',
    ];
    for (const code of codes) {
      expect(typeof OAUTH_MESSAGES[code]).toBe('string');
      expect(OAUTH_MESSAGES[code].length).toBeGreaterThan(0);
      expect(oauthMessage(code)).toBe(OAUTH_MESSAGES[code]);
    }
  });
});

describe('safeNext', () => {
  it('rejects a protocol-relative URL', () => {
    expect(safeNext('//evil.com', '/app')).toBe('/app');
  });

  it('rejects an absolute URL', () => {
    expect(safeNext('https://evil.com/app', '/app')).toBe('/app');
    expect(safeNext('http://evil.com', '/app')).toBe('/app');
  });

  it('rejects null and undefined, falling back', () => {
    expect(safeNext(null, '/app')).toBe('/app');
    expect(safeNext(undefined, '/app')).toBe('/app');
  });

  it('rejects a path that does not start with a slash', () => {
    expect(safeNext('app/x', '/app')).toBe('/app');
  });

  it('rejects an empty string', () => {
    expect(safeNext('', '/app')).toBe('/app');
  });

  it('accepts a same-origin path', () => {
    expect(safeNext('/app/x', '/app')).toBe('/app/x');
  });
});
