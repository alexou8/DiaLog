import type { Metadata } from 'next';
import Link from 'next/link';
import { redirect } from 'next/navigation';
import { getCurrentUser } from '@/lib/auth/current-user';
import { SignUpForm } from './SignUpForm';
import { AuthNotice } from '@/components/auth/AuthNotice';
import { AuthDivider, GoogleButton } from '@/components/auth/GoogleButton';
import { isGoogleEnabled } from '@/lib/auth/oauth/google';

export const metadata: Metadata = { title: 'Create your account', robots: { index: false } };

export default async function SignUpPage({
  searchParams,
}: {
  searchParams: Promise<{ error?: string }>;
}) {
  // Verified here rather than in middleware: only a database read can tell a
  // live session from a revoked one, and bouncing a revoked cookie away from
  // this page is what used to make it unreachable. See middleware.ts.
  if (await getCurrentUser()) redirect('/app');

  const params = await searchParams;
  const google = isGoogleEnabled();

  return (
    <>
      <h1 className="text-2xl font-bold sm:text-3xl">Create your account</h1>
      <p className="mt-2 text-ink-muted">
        {google
          ? 'You only need an email address, or your Google account. There is nothing to pay and no card to enter.'
          : 'You only need an email address and a password. There is nothing to pay and no card to enter.'}
      </p>

      <AuthNotice code={params.error} />

      {google ? (
        <>
          <div className="mt-8">
            <GoogleButton label="Continue with Google" className="w-full" />
          </div>
          <AuthDivider label="or sign up with email" />
        </>
      ) : null}

      <SignUpForm />

      <p className="mt-6 text-center">
        Already have an account?{' '}
        <Link href="/sign-in" className="font-semibold underline underline-offset-4">
          Sign in
        </Link>
      </p>
    </>
  );
}
