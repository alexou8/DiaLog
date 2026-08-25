import type { Metadata } from 'next';
import Link from 'next/link';
import { SignUpForm } from './SignUpForm';

export const metadata: Metadata = { title: 'Create your account', robots: { index: false } };

export default function SignUpPage() {
  return (
    <>
      <h1 className="text-2xl font-bold sm:text-3xl">Create your account</h1>
      <p className="mt-2 text-ink-muted">
        You only need an email address and a password. There is nothing to pay and no card to enter.
      </p>
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
