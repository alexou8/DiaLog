'use client';

import { useActionState } from 'react';
import { forgotPasswordAction } from '@/lib/actions/password-reset';
import type { ActionState } from '@/lib/actions/auth';
import { ActionForm, Field, SubmitButton, TextInput } from '@/components/ui/form';

export function ForgotPasswordForm() {
  const [state, action, pending] = useActionState<ActionState | null, FormData>(
    forgotPasswordAction,
    null,
  );
  return (
    <ActionForm
      action={action}
      state={state}
      pending={pending}
      className="mt-6"
      labels={{ email: 'Email address' }}
    >
      <Field label="Email address" required name="email">
        {({ id, describedBy, invalid }) => (
          <TextInput
            id={id}
            name="email"
            type="email"
            required
            autoComplete="email"
            inputMode="email"
            maxLength={254}
            aria-describedby={describedBy}
            invalid={invalid}
          />
        )}
      </Field>
      <SubmitButton className="w-full" pendingLabel="Requesting link…">
        Send reset link
      </SubmitButton>
    </ActionForm>
  );
}
