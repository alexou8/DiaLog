'use client';

import { useActionState } from 'react';
import { useFormStatus } from 'react-dom';
import { addHydrationAction, type RecordActionState } from '@/lib/actions/records';
import { Button } from '@/components/ui';
import { Field, FormStatus, RadioCards, TextInput } from '@/components/ui/form';

function Submit() {
  const { pending } = useFormStatus();
  return (
    <Button type="submit" className="w-full text-lg" disabled={pending}>
      {pending ? 'Saving…' : 'Save'}
    </Button>
  );
}

export function HydrationForm({ defaultTime }: { defaultTime: string }) {
  const [state, action] = useActionState<RecordActionState | null, FormData>(
    addHydrationAction,
    null,
  );

  return (
    <form action={action} noValidate>
      <FormStatus
        status={state && !state.ok && state.message ? { ok: false, message: state.message } : null}
      />

      <Field label="How much?" required error={state?.errors?.volume}>
        {({ id, describedBy, invalid }) => (
          <TextInput
            id={id}
            name="volume"
            type="number"
            step="any"
            inputMode="decimal"
            required
            autoFocus
            className="text-2xl"
            aria-describedby={describedBy}
            invalid={invalid}
          />
        )}
      </Field>

      <RadioCards
        name="unit"
        legend="Measured in"
        defaultValue="ML"
        columns={3}
        options={[
          { value: 'ML', label: 'Millilitres' },
          { value: 'CUP', label: 'Cups', description: 'Counted as 250 mL' },
          { value: 'FL_OZ', label: 'Fluid ounces' },
        ]}
      />

      <Field label="When?" required error={state?.errors?.takenAt}>
        {({ id, describedBy, invalid }) => (
          <TextInput
            id={id}
            name="takenAt"
            type="datetime-local"
            required
            defaultValue={defaultTime}
            aria-describedby={describedBy}
            invalid={invalid}
          />
        )}
      </Field>

      <Submit />
    </form>
  );
}
