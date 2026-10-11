'use client';

import { useActionState } from 'react';
import { signUpAction, type ActionState } from '@/lib/actions/auth';
import { ActionForm, Field, SubmitButton, TextInput } from '@/components/ui/form';

export function SignUpForm() {
  const [state, action, pending] = useActionState<ActionState | null, FormData>(signUpAction, null);

  return (
    <ActionForm
      action={action}
      state={state}
      pending={pending}
      className="mt-8"
      labels={{ displayName: 'Your name', email: 'Email address', password: 'Password' }}
    >
      <Field label="Your name" hint="Used to greet you. You can leave it blank." name="displayName">
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

      <Field label="Email address" required name="email">
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
        hint="10 to 72 characters (a little fewer with accented letters or emoji). A short phrase you will remember is stronger than a short jumble."
        name="password"
      >
        {({ id, describedBy, invalid }) => (
          <TextInput
            id={id}
            name="password"
            type="password"
            required
            autoComplete="new-password"
            minLength={10}
            // Characters, not bytes; see reset-password-form.tsx.
            maxLength={72}
            aria-describedby={describedBy}
            invalid={invalid}
          />
        )}
      </Field>

      <SubmitButton className="w-full" pendingLabel="Creating your account…">
        Create account
      </SubmitButton>
      <p className="mt-4 text-sm text-ink-muted">
        By creating an account you agree to the terms of use. DiaLog does not provide medical
        advice.
      </p>
    </ActionForm>
  );
}
