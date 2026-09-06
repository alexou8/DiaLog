'use client';

import { Button } from '@/components/ui';
import { Field, FormStatus, TextInput } from '@/components/ui/form';

export function ResetPasswordForm({ proof, error }: { proof: string; error?: string }) {
  // Client boundary only for Field's render prop. This is intentionally a
  // native form, never an Action: successful submission revokes all sessions.
  return (
    <form method="post" action="/api/auth/password/reset" className="mt-6" noValidate>
      <input type="hidden" name="proof" value={proof} />
      <FormStatus status={error ? { ok: false, message: error } : null} />
      <Field
        label="New password"
        required
        hint="Use 10–200 characters. A short phrase works well."
        error={error}
      >
        {({ id, describedBy, invalid }) => (
          <TextInput
            id={id}
            name="newPassword"
            type="password"
            autoComplete="new-password"
            required
            maxLength={200}
            aria-describedby={describedBy}
            invalid={invalid}
          />
        )}
      </Field>
      <Field label="Confirm new password" required>
        {({ id, describedBy, invalid }) => (
          <TextInput
            id={id}
            name="confirmPassword"
            type="password"
            autoComplete="new-password"
            required
            maxLength={200}
            aria-describedby={describedBy}
            invalid={invalid}
          />
        )}
      </Field>
      <Button type="submit" className="w-full">
        Reset password
      </Button>
    </form>
  );
}
