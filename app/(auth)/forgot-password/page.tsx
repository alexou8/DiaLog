import type { Metadata } from 'next';
import Link from 'next/link';
import { ForgotPasswordForm } from './forgot-password-form';

export const metadata: Metadata = {
  title: 'Forgot your password',
  robots: { index: false },
  referrer: 'no-referrer',
};
export const dynamic = 'force-dynamic';

export default function ForgotPasswordPage() {
  // Never gate recovery on a cookie: revoked/unverifiable cookies previously
  // locked people out of the very page needed to restore access.
  return (
    <>
      <h1 className="text-2xl font-bold sm:text-3xl">Forgot your password?</h1>
      <p className="mt-2 text-ink-muted">
        Enter your account email to request a link. Links expire after 30 minutes. Google-only
        accounts should continue with Google sign-in.
      </p>
      <ForgotPasswordForm />
      <p className="mt-6">
        <Link href="/sign-in" className="font-semibold underline underline-offset-4">
          Back to sign in
        </Link>
      </p>
    </>
  );
}
