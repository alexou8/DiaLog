'use client';

import { useActionState } from 'react';
import { useFormStatus } from 'react-dom';
import { signInAction, type ActionState } from '@/lib/actions/auth';
import { Button } from '@/components/ui';
import { Field, FormStatus, TextInput } from '@/components/ui/form';

function Submit() {
  const { pending } = useFormStatus();
  return (
    <Button type="submit" className="w-full" disabled={pending}>
      {pending ? 'Signing in…' : 'Sign in'}
    </Button>
  );
}

export function SignInForm() {
  const [state, action] = useActionState<ActionState | null, FormData>(signInAction, null);

  return (
    <form action={action} className="mt-8" noValidate>
      <FormStatus status={state && !state.ok && state.message ? { ok: false, message: state.message } : null} />

      <Field label="Email address" required error={state?.errors?.email}>
        {({ id, describedBy, invalid }) => (
          <TextInput
            id={id}
            name="email"
            type="email"
            required
            autoComplete="email"
            inputMode="email"
            aria-describedby={describedBy}
            invalid={invalid}
          />
        )}
      </Field>

      <Field label="Password" required error={state?.errors?.password}>
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

      <Submit />
    </form>
  );
}
