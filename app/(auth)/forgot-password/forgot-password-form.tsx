'use client';

import { useActionState } from 'react';
import { useFormStatus } from 'react-dom';
import { forgotPasswordAction } from '@/lib/actions/password-reset';
import type { ActionState } from '@/lib/actions/auth';
import { Button } from '@/components/ui';
import { Field, FormStatus, TextInput } from '@/components/ui/form';

function Submit() {
  const { pending } = useFormStatus();
  return (
    <Button type="submit" className="w-full" disabled={pending}>
      {pending ? 'Requesting link…' : 'Send reset link'}
    </Button>
  );
}

export function ForgotPasswordForm() {
  const [state, action] = useActionState<ActionState | null, FormData>(forgotPasswordAction, null);
  return (
    <form action={action} className="mt-6" noValidate>
      <FormStatus status={state?.message ? { ok: state.ok, message: state.message } : null} />
      <Field label="Email address" required error={state?.errors?.email}>
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
      <Submit />
    </form>
  );
}
