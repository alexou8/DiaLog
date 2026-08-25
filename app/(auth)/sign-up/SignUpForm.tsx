'use client';

import { useActionState } from 'react';
import { useFormStatus } from 'react-dom';
import { signUpAction, type ActionState } from '@/lib/actions/auth';
import { Button } from '@/components/ui';
import { Field, FormStatus, TextInput } from '@/components/ui/form';

function Submit() {
  const { pending } = useFormStatus();
  return (
    <Button type="submit" className="w-full" disabled={pending}>
      {pending ? 'Creating your account…' : 'Create account'}
    </Button>
  );
}

export function SignUpForm() {
  const [state, action] = useActionState<ActionState | null, FormData>(signUpAction, null);

  return (
    <form action={action} className="mt-8" noValidate>
      <FormStatus
        status={state && !state.ok && state.message ? { ok: false, message: state.message } : null}
      />

      <Field
        label="Your name"
        hint="Used to greet you. You can leave it blank."
        error={state?.errors?.displayName}
      >
        {({ id, describedBy, invalid }) => (
          <TextInput
            id={id}
            name="displayName"
            autoComplete="name"
            aria-describedby={describedBy}
            invalid={invalid}
          />
        )}
      </Field>

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

      <Field
        label="Password"
        required
        hint="At least 10 characters. A short phrase you will remember is stronger than a short jumble."
        error={state?.errors?.password}
      >
        {({ id, describedBy, invalid }) => (
          <TextInput
            id={id}
            name="password"
            type="password"
            required
            autoComplete="new-password"
            minLength={10}
            aria-describedby={describedBy}
            invalid={invalid}
          />
        )}
      </Field>

      <Submit />
      <p className="mt-4 text-sm text-ink-muted">
        By creating an account you agree to the terms of use. DiaLog does not provide medical
        advice.
      </p>
    </form>
  );
}
