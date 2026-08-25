'use client';

import { useActionState } from 'react';
import { useFormStatus } from 'react-dom';
import {
  changePasswordAction,
  deleteAccountAction,
  deleteAllRecordsAction,
  signOutEverywhereAction,
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

export function ChangePasswordForm() {
  const [state, action] = useActionState<ActionState | null, FormData>(changePasswordAction, null);

  return (
    <form
      action={action}
      noValidate
      className="rounded-[var(--radius-card)] border border-line bg-surface p-5 sm:p-6"
    >
      <h3 className="text-lg font-semibold">Change your password</h3>
      <p className="mt-1 text-sm text-ink-muted">
        Changing your password signs every other device out. This device stays signed in.
      </p>

      <div className="mt-4">
        <FormStatus
          status={state && state.message ? { ok: state.ok, message: state.message } : null}
        />
      </div>

      <Field label="Current password" required error={state?.errors?.currentPassword}>
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

      <Field
        label="New password"
        required
        hint="At least 10 characters. A short phrase works well."
        error={state?.errors?.newPassword}
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

      <Field label="Confirm new password" required error={state?.errors?.confirmPassword}>
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

      <SubmitButton pendingLabel="Changing…">Change password</SubmitButton>
    </form>
  );
}

// ------------------------------------------------------------- sign out all

export function SignOutEverywhereForm() {
  const [state, action] = useActionState<ActionState | null, FormData>(
    signOutEverywhereAction,
    null,
  );

  return (
    <form
      action={action}
      className="rounded-[var(--radius-card)] border border-line bg-surface p-5 sm:p-6"
    >
      <h3 className="text-lg font-semibold">Sign out everywhere</h3>
      <p className="mt-1 text-sm text-ink-muted">
        Immediately signs out every device and browser signed in to your account, except this one.
      </p>
      <div className="mt-4">
        <FormStatus
          status={state && state.message ? { ok: state.ok, message: state.message } : null}
        />
      </div>
      <SubmitButton pendingLabel="Signing out other devices…" variant="secondary">
        Sign out everywhere else
      </SubmitButton>
    </form>
  );
}

// -------------------------------------------------------- destructive zone

export function DeleteAllRecordsForm({
  totalRecords,
  userEmail,
}: {
  totalRecords: number;
  userEmail: string;
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
        <Field label="Your password" required error={state?.errors?.password}>
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
      </div>

      <SubmitButton pendingLabel="Deleting all records…" variant="danger">
        Permanently delete all records
      </SubmitButton>
    </form>
  );
}

export function DeleteAccountForm({ userEmail }: { userEmail: string }) {
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
        <Field label="Your password" required error={state?.errors?.password}>
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
      </div>

      <SubmitButton pendingLabel="Deleting your account…" variant="danger">
        Permanently delete my account
      </SubmitButton>
    </form>
  );
}
