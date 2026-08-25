/**
 * Account-resolution rules for federated sign-in.
 *
 * Kept as pure functions over an already-fetched snapshot of the database so
 * the policy — especially the email-collision cases — can be unit tested
 * without a database, and so the route handler reads as a single decision.
 *
 * The governing rule: DiaLog never merges a Google identity into an existing
 * password account on the strength of a matching email alone. This is health
 * data; whoever controls the Google account would silently inherit the whole
 * record. Linking always requires proving control of the DiaLog account first
 * by signing in with the password, then linking from Settings.
 */
import type { GoogleIdentity } from './google';

export type OAuthErrorCode =
  | 'unverified_email'
  | 'email_in_use'
  | 'linked_elsewhere'
  | 'provider_already_linked'
  | 'not_configured'
  | 'invalid_state'
  | 'access_denied'
  | 'exchange_failed';

/** What the database knows before we decide. */
export interface SignInSnapshot {
  /** User id already linked to this Google subject, if any. */
  identityUserId: string | null;
  /** A DiaLog account holding the Google-asserted email, if any. */
  userWithEmail: { id: string; hasPassword: boolean } | null;
}

export type SignInOutcome =
  | { kind: 'sign_in'; userId: string }
  | { kind: 'create' }
  | { kind: 'blocked'; code: OAuthErrorCode; email?: string };

export function resolveGoogleSignIn(
  identity: GoogleIdentity,
  snapshot: SignInSnapshot,
): SignInOutcome {
  // An unverified Google address proves nothing about who owns that mailbox.
  if (!identity.emailVerified) return { kind: 'blocked', code: 'unverified_email' };

  // Known subject: sign in, regardless of what the email says today. Google
  // subjects are stable and never reused, so this survives the user renaming
  // their Google address.
  if (snapshot.identityUserId) return { kind: 'sign_in', userId: snapshot.identityUserId };

  // The collision case. Includes accounts with no password (linked to some
  // other provider): either way this Google account is not yet proven to
  // belong to that person.
  if (snapshot.userWithEmail) {
    return { kind: 'blocked', code: 'email_in_use', email: identity.email };
  }

  return { kind: 'create' };
}

/** What the database knows when a signed-in user adds Google. */
export interface LinkSnapshot {
  identityUserId: string | null;
  currentUserId: string;
  currentUserHasGoogle: boolean;
}

export type LinkOutcome =
  | { kind: 'link' }
  | { kind: 'already_linked' }
  | { kind: 'blocked'; code: OAuthErrorCode };

export function resolveGoogleLink(identity: GoogleIdentity, snapshot: LinkSnapshot): LinkOutcome {
  if (!identity.emailVerified) return { kind: 'blocked', code: 'unverified_email' };
  if (snapshot.identityUserId === snapshot.currentUserId) return { kind: 'already_linked' };
  if (snapshot.identityUserId) return { kind: 'blocked', code: 'linked_elsewhere' };
  if (snapshot.currentUserHasGoogle) return { kind: 'blocked', code: 'provider_already_linked' };
  return { kind: 'link' };
}

/**
 * Plain-language explanations, shown on the sign-in page or Settings. Each one
 * names the next thing to do, because "something went wrong" leaves a person
 * locked out of their own records with nowhere to go.
 */
export const OAUTH_MESSAGES: Record<OAuthErrorCode, string> = {
  unverified_email:
    'Google has not verified the email address on that account, so we cannot use it to sign in. Please verify it with Google, or sign in with your email and password.',
  email_in_use:
    'You already have a DiaLog account with that email address and a password. Sign in with your password below, then connect Google from Settings — that way nobody can reach your records just by holding the Google account.',
  linked_elsewhere:
    'That Google account is already connected to a different DiaLog account. Disconnect it there first, or use another Google account.',
  provider_already_linked:
    'Your account is already connected to a Google account. Disconnect the current one before connecting a different one.',
  not_configured: 'Google sign-in is not available on this deployment. Please use your password.',
  invalid_state:
    'That sign-in link expired or did not come from this browser. Please try signing in again.',
  access_denied: 'Google sign-in was cancelled. You can try again or use your email and password.',
  exchange_failed:
    'We could not complete sign-in with Google just now. Please try again in a moment, or use your email and password.',
};

export function oauthMessage(code: string | null | undefined): string | null {
  if (!code) return null;
  return OAUTH_MESSAGES[code as OAuthErrorCode] ?? null;
}
