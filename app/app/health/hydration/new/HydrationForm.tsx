'use client';

import { useActionState } from 'react';
import { addHydrationAction, type RecordActionState } from '@/lib/actions/records';
import { ActionForm, Field, RadioCards, SubmitButton, TextInput } from '@/components/ui/form';

export function HydrationForm({ defaultTime }: { defaultTime: string }) {
  const [state, action, pending] = useActionState<RecordActionState | null, FormData>(
    addHydrationAction,
    null,
  );

  return (
    <ActionForm
      action={action}
      state={state}
      pending={pending}
      labels={{
        volume: 'How much?',
        takenAt: 'When?',
        unit: 'Measured in',
      }}
    >
      <Field label="How much?" required name="volume">
        {({ id, describedBy, invalid }) => (
          <TextInput
            id={id}
            name="volume"
            type="number"
            step="any"
            inputMode="decimal"
            required
            autoFocus={!state}
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

      <Field label="When?" required name="takenAt">
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

      <SubmitButton className="w-full text-lg">Save</SubmitButton>
    </ActionForm>
  );
}
