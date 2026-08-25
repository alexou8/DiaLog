import { GoogleButton } from '@/components/auth/GoogleButton';
import { Button } from '@/components/ui';

/**
 * The Google connection panel. A server component: every outcome of connecting
 * or disconnecting is a page-level notice carried on the query string, so
 * nothing here needs to hold state on the client.
 */
export function ConnectedAccounts({
  googleEmail,
  hasPassword,
  notice,
}: {
  googleEmail: string | null;
  hasPassword: boolean;
  notice: { ok: boolean; message: string } | null;
}) {
  return (
    <div className="rounded-[var(--radius-card)] border border-line bg-surface p-5 sm:p-6">
      <h3 className="text-lg font-semibold">Sign in with Google</h3>
      <p className="mt-1 max-w-prose text-sm text-ink-muted">
        {googleEmail
          ? `Your account is connected to ${googleEmail}. DiaLog only uses Google to check it is you — none of your health data is shared with Google.`
          : 'Connect your Google account so you can sign in with one tap. DiaLog only uses Google to check it is you — none of your health data is shared with Google.'}
      </p>

      {notice ? (
        <p
          role="status"
          className={
            notice.ok
              ? 'mt-4 rounded-xl border border-positive/40 bg-positive-soft p-3 text-sm font-medium text-positive'
              : 'mt-4 rounded-xl border border-critical/40 bg-critical-soft p-3 text-sm font-medium text-critical'
          }
        >
          <span aria-hidden="true">{notice.ok ? '✓ ' : '⚠ '}</span>
          {notice.message}
        </p>
      ) : null}

      <div className="mt-4">
        {googleEmail ? (
          <form method="post" action="/api/auth/google/disconnect">
            <Button type="submit" variant="secondary">
              Disconnect Google
            </Button>
            {!hasPassword ? (
              <p className="mt-2 text-sm text-ink-muted">
                Google is currently the only way in to this account. Set a password above before
                disconnecting it.
              </p>
            ) : null}
          </form>
        ) : (
          <GoogleButton mode="link" label="Connect Google" />
        )}
      </div>
    </div>
  );
}
