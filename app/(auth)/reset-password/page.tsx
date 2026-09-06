import type { Metadata } from 'next';
import Link from 'next/link';
import { Callout } from '@/components/ui';
import { isResetProofUsable, RECOVERY_UNAVAILABLE } from '@/lib/auth/password-reset';
import { PASSWORD_POLICY_MESSAGES } from '@/lib/auth/password';
import { isSessionSecretConfigured } from '@/lib/auth/session';
import type { ResetOutcome } from '@/app/api/auth/password/reset/route';
import { ResetPasswordForm } from './reset-password-form';

export const metadata: Metadata = {
  title: 'Reset your password',
  robots: { index: false },
  // Deliberately 'strict-origin' rather than 'no-referrer', which is what the
  // forgot-password page uses. Chrome derives a form submission's Origin header
  // from the document's referrer policy, so 'no-referrer' made this page send
  // `Origin: null` — and the reset route requires Origin, so it rejected every
  // legitimate submission as cross_origin and recovery could not complete at
  // all. 'strict-origin' sends the bare origin and never the path, so the proof
  // in this page's query string still never leaves in a Referer header.
  referrer: 'strict-origin',
};
export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

const FEEDBACK: Record<ResetOutcome, string> = {
  ...PASSWORD_POLICY_MESSAGES,
  mismatch: 'The passwords do not match. Please enter them again.',
  invalid: 'This reset link is invalid or has expired. Please request a new one.',
  rate_limited: 'Too many reset attempts. Please try again later.',
  unavailable: RECOVERY_UNAVAILABLE,
};

export default async function ResetPasswordPage({
  searchParams,
}: {
  searchParams: Promise<{ proof?: string; password?: string }>;
}) {
  const params = await searchParams;
  let proof: string | null = null;
  let error =
    params.password && Object.hasOwn(FEEDBACK, params.password)
      ? FEEDBACK[params.password as ResetOutcome]
      : undefined;
  try {
    if (!isSessionSecretConfigured()) error = RECOVERY_UNAVAILABLE;
    else if (typeof params.proof === 'string' && (await isResetProofUsable(params.proof)))
      proof = params.proof;
  } catch {
    error = RECOVERY_UNAVAILABLE;
  }
  return (
    <>
      <h1 className="text-2xl font-bold sm:text-3xl">Reset your password</h1>
      {proof ? (
        <>
          <p className="mt-2 text-ink-muted">
            Choose a new password. All devices will be signed out, and you will need to sign in
            again.
          </p>
          <ResetPasswordForm proof={proof} error={error} />
        </>
      ) : (
        <Callout tone="critical" icon="caution" role="alert">
          {error ?? FEEDBACK.invalid}
        </Callout>
      )}
      <p className="mt-6">
        <Link href="/forgot-password" className="font-semibold underline underline-offset-4">
          Request a new reset link
        </Link>
      </p>
      <p className="mt-4">
        <Link href="/sign-in" className="font-semibold underline underline-offset-4">
          Back to sign in
        </Link>
      </p>
    </>
  );
}
