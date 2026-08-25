'use client';

import { useActionState } from 'react';
import { useFormStatus } from 'react-dom';
import { unlinkGoogleAction, type ActionState } from '@/lib/actions/preferences';
import { Button } from '@/components/ui';
import { FormStatus } from '@/components/ui/form';
import { GoogleButton } from '@/components/auth/GoogleButton';

function Unlink() {
  const { pending } = useFormStatus();
  return (
    <Button type="submit" variant="secondary" disabled={pending}>
      {pending ? 'Disconnecting…' : 'Disconnect Google'}
    </Button>
  );
}

export function ConnectedAccounts({
  googleEmail,
  hasPassword,
  notice,
}: {
  googleEmail: string | null;
  hasPassword: boolean;
  notice: { ok: boolean; message: string } | null;
}) {
  const [state, action] = useActionState<ActionState | null, FormData>(unlinkGoogleAction, null);
  const status = state?.message ? { ok: state.ok, message: state.message } : notice;

  return (
    <div className="rounded-[var(--radius-card)] border border-line bg-surface p-5 sm:p-6">
      <h3 className="text-lg font-semibold">Sign in with Google</h3>
      <p className="mt-1 max-w-prose text-sm text-ink-muted">
        {googleEmail
          ? `Your account is connected to ${googleEmail}. DiaLog only uses Google to check it is you — none of your health data is shared with Google.`
          : 'Connect your Google account so you can sign in with one tap. DiaLog only uses Google to check it is you — none of your health data is shared with Google.'}
      </p>

      <div className="mt-4">
        <FormStatus status={status} />
      </div>

      {googleEmail ? (
        <form action={action}>
          <Unlink />
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
  );
}
