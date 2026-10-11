'use client';

import { useActionState } from 'react';
import { addWeightAction, type RecordActionState } from '@/lib/actions/records';
import { toLocalInputValue } from '@/lib/domain/time';
import { ActionForm, Field, Select, SubmitButton, TextInput } from '@/components/ui/form';

export function WeightForm({ timezone }: { timezone: string }) {
  const [state, action, pending] = useActionState<RecordActionState | null, FormData>(
    addWeightAction,
    null,
  );
  const now = toLocalInputValue(new Date(), timezone);

  return (
    <ActionForm
      action={action}
      state={state}
      pending={pending}
      labels={{
        weight: 'Weight',
        takenAt: 'When',
        note: 'Note',
      }}
    >
      <div className="grid grid-cols-[1fr_auto] items-start gap-3">
        <Field label="Weight" required name="weight">
          {({ id, describedBy, invalid }) => (
            <TextInput
              id={id}
              name="weight"
              type="number"
              required
              min={0}
              step="any"
              inputMode="decimal"
              aria-describedby={describedBy}
              invalid={invalid}
            />
          )}
        </Field>
        <Field label="Unit" htmlFor="unit">
          {({ id }) => (
            <Select id={id} name="unit" defaultValue="KG">
              <option value="KG">kg</option>
              <option value="LB">lb</option>
            </Select>
          )}
        </Field>
      </div>

      <Field label="When" required name="takenAt">
        {({ id, describedBy, invalid }) => (
          <TextInput
            id={id}
            name="takenAt"
            type="datetime-local"
            required
            defaultValue={now}
            aria-describedby={describedBy}
            invalid={invalid}
          />
        )}
      </Field>

      <Field label="Note" name="note">
        {({ id, describedBy, invalid }) => (
          <TextInput
            id={id}
            name="note"
            maxLength={500}
            aria-describedby={describedBy}
            invalid={invalid}
          />
        )}
      </Field>

      <SubmitButton className="w-full sm:w-auto">Save weight</SubmitButton>
    </ActionForm>
  );
}
