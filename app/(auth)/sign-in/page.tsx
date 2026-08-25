import type { Metadata } from 'next';
import Link from 'next/link';
import { redirect } from 'next/navigation';
import { getCurrentUser } from '@/lib/auth/current-user';
import { SignInForm } from './SignInForm';
import { AuthNotice } from '@/components/auth/AuthNotice';
import { AuthDivider, GoogleButton } from '@/components/auth/GoogleButton';
import { isGoogleEnabled } from '@/lib/auth/oauth/google';

export const metadata: Metadata = { title: 'Sign in', robots: { index: false } };

export default async function SignInPage({
  searchParams,
}: {
  searchParams: Promise<{ error?: string; email?: string; next?: string }>;
}) {
  // Verified here rather than in middleware: only a database read can tell a
  // live session from a revoked one, and bouncing a revoked cookie away from
  // this page is what used to make it unreachable. See middleware.ts.
  if (await getCurrentUser()) redirect('/app');

  const params = await searchParams;

  return (
    <>
      <h1 className="text-2xl font-bold sm:text-3xl">Welcome back</h1>
      <p className="mt-2 text-ink-muted">
        Sign in to see your readings and this week&apos;s observations.
      </p>

      <AuthNotice code={params.error} />

      <SignInForm defaultEmail={params.email} />

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
