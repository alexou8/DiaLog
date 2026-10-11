'use client';

import { useActionState } from 'react';
import { signInAction, type ActionState } from '@/lib/actions/auth';
import { ActionForm, Field, SubmitButton, TextInput } from '@/components/ui/form';

export function SignInForm({ defaultEmail, next }: { defaultEmail?: string; next?: string }) {
  const [state, action, pending] = useActionState<ActionState | null, FormData>(signInAction, null);

  return (
    <ActionForm
      action={action}
      state={state}
      pending={pending}
      className="mt-6"
      labels={{ email: 'Email address', password: 'Password' }}
    >
      {/* Where to land after signing in: a deep link, or a PWA shortcut such
          as "Add a glucose reading". The Google button already forwarded it;
          email sign-in dropped it and always landed on Home. The value is
          only ever a hint: the server re-checks it is a same-origin path. */}
      {next ? <input type="hidden" name="next" value={next} /> : null}

      <Field label="Email address" required name="email">
        {({ id, describedBy, invalid }) => (
          <TextInput
            id={id}
            name="email"
            type="email"
            required
            defaultValue={defaultEmail}
            autoComplete="email"
            inputMode="email"
            aria-describedby={describedBy}
            invalid={invalid}
          />
        )}
      </Field>

      <Field label="Password" required name="password">
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

      <SubmitButton className="w-full" pendingLabel="Signing in…">
        Sign in
      </SubmitButton>
    </ActionForm>
  );
}
