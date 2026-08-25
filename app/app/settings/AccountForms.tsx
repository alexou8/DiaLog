'use client';

import { useActionState } from 'react';
import { useFormStatus } from 'react-dom';
import {
  deleteAccountAction,
  deleteAllRecordsAction,
  type ActionState,
} from '@/lib/actions/preferences';
import { Button } from '@/components/ui';
import { Field, FormStatus, TextInput } from '@/components/ui/form';

function SubmitButton({
  children,
  pendingLabel,
  variant = 'primary',
}: {
  children: React.ReactNode;
  pendingLabel: string;
  variant?: 'primary' | 'danger' | 'secondary';
}) {
  const { pending } = useFormStatus();
  return (
    <Button type="submit" variant={variant} disabled={pending}>
      {pending ? pendingLabel : children}
    </Button>
  );
}

// ---------------------------------------------------------------- password

/**
 * The result of a password or session change, as rendered on the page after
 * the browser has followed the handler's redirect.
 *
 * `field` places the message on the input it is about, so a refusal reads the
 * same way it did when this was a Server Action returning field errors — the
 * delivery mechanism changed, the accessible behaviour did not.
 */
export interface AccountFeedback {
  ok: boolean;
  message: string;
  field?: 'currentPassword' | 'newPassword' | 'confirmPassword';
}

function fieldError(feedback: AccountFeedback | null, field: AccountFeedback['field']) {
  return feedback && feedback.field === field ? feedback.message : undefined;
}

/**
 * Set or change the account password.
 *
 * A plain `method="post"` form, deliberately: it posts to a route handler and
 * the browser performs the navigation itself. Changing a password revokes this
 * browser's session cookie, and a response carrying that revocation cannot be
 * delivered through the client router — see lib/auth/route-form.ts. This form
 * must therefore never be converted back to `useActionState`.
 */
export function ChangePasswordForm({
  hasPassword = true,
  feedback = null,
}: {
  hasPassword?: boolean;
  feedback?: AccountFeedback | null;
}) {
  return (
    <form
      method="post"
      action="/api/auth/password"
      noValidate
      className="rounded-[var(--radius-card)] border border-line bg-surface p-5 sm:p-6"
    >
      <h3 className="text-lg font-semibold">
        {hasPassword ? 'Change your password' : 'Set a password'}
      </h3>
      <p className="mt-1 text-sm text-ink-muted">
        {hasPassword
          ? 'Changing your password signs every other device out. This device stays signed in.'
          : 'You currently sign in with Google. Setting a password gives you a second way in, so you are not locked out if you lose access to that Google account.'}
      </p>

      <div className="mt-4">
        <FormStatus
          status={
            feedback && !feedback.field ? { ok: feedback.ok, message: feedback.message } : null
          }
        />
      </div>

      {hasPassword ? (
        <Field label="Current password" required error={fieldError(feedback, 'currentPassword')}>
          {({ id, describedBy, invalid }) => (
            <TextInput
              id={id}
              name="currentPassword"
              type="password"
              required
              autoComplete="current-password"
              aria-describedby={describedBy}
              invalid={invalid}
            />
          )}
        </Field>
      ) : null}

      <Field
        label="New password"
        required
        hint="At least 10 characters. A short phrase works well."
        error={fieldError(feedback, 'newPassword')}
      >
        {({ id, describedBy, invalid }) => (
          <TextInput
            id={id}
            name="newPassword"
            type="password"
            required
            autoComplete="new-password"
            aria-describedby={describedBy}
            invalid={invalid}
          />
        )}
      </Field>

      <Field
        label={hasPassword ? 'Confirm new password' : 'Confirm password'}
        required
        error={fieldError(feedback, 'confirmPassword')}
      >
        {({ id, describedBy, invalid }) => (
          <TextInput
            id={id}
            name="confirmPassword"
            type="password"
            required
            autoComplete="new-password"
            aria-describedby={describedBy}
            invalid={invalid}
          />
        )}
      </Field>

      <Button type="submit" variant="primary">
        {hasPassword ? 'Change password' : 'Set password'}
      </Button>
    </form>
  );
}

// ------------------------------------------------------------- sign out all

/** Also a plain form posting to a route handler, and for the same reason. */
export function SignOutEverywhereForm({ feedback = null }: { feedback?: AccountFeedback | null }) {
  return (
    <form
      method="post"
      action="/api/auth/sessions/revoke"
      className="rounded-[var(--radius-card)] border border-line bg-surface p-5 sm:p-6"
    >
      <h3 className="text-lg font-semibold">Sign out everywhere</h3>
      <p className="mt-1 text-sm text-ink-muted">
        Immediately signs out every device and browser signed in to your account, except this one.
      </p>
      <div className="mt-4">
        <FormStatus status={feedback ? { ok: feedback.ok, message: feedback.message } : null} />
      </div>
      <Button type="submit" variant="secondary">
        Sign out everywhere else
      </Button>
    </form>
  );
}

/**
 * The deliberate-action check on a destructive form. Accounts that sign in with
 * Google alone have no password to re-enter, so they type a fixed phrase
 * instead — see `checkDestructiveConfirmation` in lib/actions/preferences.ts.
 */
function ConfirmSecret({
  hasPassword,
  errors,
}: {
  hasPassword: boolean;
  errors?: Record<string, string>;
}) {
  if (!hasPassword) {
    return (
      <Field label="Type DELETE to confirm" required error={errors?.confirmPhrase}>
        {({ id, describedBy, invalid }) => (
          <TextInput
            id={id}
            name="confirmPhrase"
            required
            autoComplete="off"
            aria-describedby={describedBy}
            invalid={invalid}
          />
        )}
      </Field>
    );
  }
  return (
    <Field label="Your password" required error={errors?.password}>
      {({ id, describedBy, invalid }) => (
        <TextInput
          id={id}
          name="password"
          type="password"
          required
          autoComplete="current-password"
          aria-describedby={describedBy}
          invalid={invalid}
        />
      )}
    </Field>
  );
}

// -------------------------------------------------------- destructive zone

export function DeleteAllRecordsForm({
  totalRecords,
  userEmail,
  hasPassword = true,
}: {
  totalRecords: number;
  userEmail: string;
  hasPassword?: boolean;
}) {
  const [state, action] = useActionState<ActionState | null, FormData>(
    deleteAllRecordsAction,
    null,
  );

  return (
    <form action={action} noValidate aria-labelledby="delete-records-heading">
      <h4 id="delete-records-heading" className="font-semibold text-critical">
        Delete all health records
      </h4>
      <p className="mt-1 max-w-prose text-sm text-ink">
        Permanently deletes all {totalRecords.toLocaleString()} of your glucose readings, meals,
        activity, sleep, medication, weight, blood pressure, mood and note entries. Your account,
        sign-in and settings are kept. This cannot be undone — there is no way to recover deleted
        records.
      </p>

      <div className="mt-4">
        <FormStatus
          status={state && state.message ? { ok: state.ok, message: state.message } : null}
        />
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <Field
          label="Type your account email to confirm"
          required
          error={state?.errors?.confirmEmail}
          hint={userEmail}
        >
          {({ id, describedBy, invalid }) => (
            <TextInput
              id={id}
              name="confirmEmail"
              type="email"
              required
              autoComplete="off"
              aria-describedby={describedBy}
              invalid={invalid}
            />
          )}
        </Field>
        <ConfirmSecret hasPassword={hasPassword} errors={state?.errors} />
      </div>

      <SubmitButton pendingLabel="Deleting all records…" variant="danger">
        Permanently delete all records
      </SubmitButton>
    </form>
  );
}

export function DeleteAccountForm({
  userEmail,
  hasPassword = true,
}: {
  userEmail: string;
  hasPassword?: boolean;
}) {
  const [state, action] = useActionState<ActionState | null, FormData>(deleteAccountAction, null);

  return (
    <form action={action} noValidate aria-labelledby="delete-account-heading">
      <h4 id="delete-account-heading" className="font-semibold text-critical">
        Delete your account
      </h4>
      <p className="mt-1 max-w-prose text-sm text-ink">
        Permanently deletes your DiaLog account and every record in it — readings, meals, activity,
        sleep, medication, weight, blood pressure, mood, notes, imports, insights and assistant
        conversations. You will be signed out immediately. This cannot be undone; there is no
        recovery, grace period, or way to reactivate the account afterwards.
      </p>

      <div className="mt-4">
        <FormStatus
          status={
            state && !state.ok && state.message ? { ok: false, message: state.message } : null
          }
        />
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <Field
          label="Type your account email to confirm"
          required
          error={state?.errors?.confirmEmail}
          hint={userEmail}
        >
          {({ id, describedBy, invalid }) => (
            <TextInput
              id={id}
              name="confirmEmail"
              type="email"
              required
              autoComplete="off"
              aria-describedby={describedBy}
              invalid={invalid}
            />
          )}
        </Field>
        <ConfirmSecret hasPassword={hasPassword} errors={state?.errors} />
      </div>

      <SubmitButton pendingLabel="Deleting your account…" variant="danger">
        Permanently delete my account
      </SubmitButton>
    </form>
  );
}
