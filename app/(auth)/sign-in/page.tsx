import type { Metadata } from 'next';
import Link from 'next/link';
import { redirect } from 'next/navigation';
import { getCurrentUser } from '@/lib/auth/current-user';
import { SignInForm } from './SignInForm';
import { AuthNotice } from '@/components/auth/AuthNotice';
import { AuthDivider, GoogleButton } from '@/components/auth/GoogleButton';
import { isGoogleEnabled } from '@/lib/auth/oauth/google';
import { safeNext } from '@/lib/auth/oauth/state';
import { Callout } from '@/components/ui';

export const metadata: Metadata = { title: 'Sign in', robots: { index: false } };

export default async function SignInPage({
  searchParams,
}: {
  searchParams: Promise<{
    error?: string;
    email?: string;
    next?: string;
    password?: string;
    unlinked?: string;
  }>;
}) {
  // Verified here rather than in middleware: only a database read can tell a
  // live session from a revoked one, and bouncing a revoked cookie away from
  // this page is what used to make it unreachable. See middleware.ts.
  if (await getCurrentUser()) redirect('/app');

  const params = await searchParams;
  // Same-origin path or nothing; the Google button applies the same check.
  const next = params.next ? safeNext(params.next, '/app') : '/app';

  return (
    <>
      <h1 className="text-2xl font-bold sm:text-3xl">Welcome back</h1>
      <p className="mt-2 text-ink-muted">
        Sign in to see your readings and this week&apos;s observations.
      </p>

      <AuthNotice code={params.error} />

      {params.password === 'reset' ? (
        <Callout tone="positive" icon="ok" role="status">
          Your password has been reset. Sign in with your new password.
        </Callout>
      ) : null}

      {params.unlinked === 'google' ? (
        // Set by a password reset that removed a linked Google identity: the
        // reset proves control of the email, not of the Google account, so
        // the link is dropped. Saying so stops a later "Sign in with Google"
        // from looking like a lost account.
        <Callout tone="info" icon="info" role="status">
          Your Google sign-in was disconnected for safety. You can reconnect it in Settings.
        </Callout>
      ) : null}

      <SignInForm defaultEmail={params.email} next={next === '/app' ? undefined : next} />

      <p className="mt-4 text-center">
        <Link href="/forgot-password" className="font-semibold underline underline-offset-4">
          Forgot your password?
        </Link>
      </p>

      {isGoogleEnabled() ? (
        <>
          <AuthDivider />
          <GoogleButton label="Sign in with Google" next={params.next} className="w-full" />
        </>
      ) : null}

      <p className="mt-6 text-center">
        New to DiaLog?{' '}
        <Link href="/sign-up" className="font-semibold underline underline-offset-4">
          Create an account
        </Link>
      </p>
    </>
  );
}
