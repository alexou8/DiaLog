import type { Metadata } from 'next';
import Link from 'next/link';
import { SignInForm } from './SignInForm';

export const metadata: Metadata = { title: 'Sign in', robots: { index: false } };

export default function SignInPage() {
  return (
    <>
      <h1 className="text-2xl font-bold sm:text-3xl">Welcome back</h1>
      <p className="mt-2 text-ink-muted">
        Sign in to see your readings and this week&apos;s observations.
      </p>
      <SignInForm />
      <p className="mt-6 text-center">
        New to DiaLog?{' '}
        <Link href="/sign-up" className="font-semibold underline underline-offset-4">
          Create an account
        </Link>
      </p>
    </>
  );
}
