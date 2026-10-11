'use client';

import { useActionState } from 'react';
import { addBloodPressureAction, type RecordActionState } from '@/lib/actions/records';
import { toLocalInputValue } from '@/lib/domain/time';
import { ActionForm, Field, SubmitButton, TextInput } from '@/components/ui/form';

export function BloodPressureForm({ timezone }: { timezone: string }) {
  const [state, action, pending] = useActionState<RecordActionState | null, FormData>(
    addBloodPressureAction,
    null,
  );
  const now = toLocalInputValue(new Date(), timezone);

  return (
    <ActionForm
      action={action}
      state={state}
      pending={pending}
      labels={{
        systolic: 'Upper number (systolic)',
        diastolic: 'Lower number (diastolic)',
        pulse: 'Pulse',
        takenAt: 'When',
        note: 'Note',
      }}
    >
      <div className="grid gap-4 sm:grid-cols-2">
        <Field label="Upper number (systolic)" required name="systolic">
          {({ id, describedBy, invalid }) => (
            <TextInput
              id={id}
              name="systolic"
              type="number"
              required
              min={50}
              max={300}
              step={1}
              inputMode="numeric"
              aria-describedby={describedBy}
              invalid={invalid}
            />
          )}
        </Field>
        <Field label="Lower number (diastolic)" required name="diastolic">
          {({ id, describedBy, invalid }) => (
            <TextInput
              id={id}
              name="diastolic"
              type="number"
              required
              min={30}
              max={200}
              step={1}
              inputMode="numeric"
              aria-describedby={describedBy}
              invalid={invalid}
            />
          )}
        </Field>
      </div>

      <Field label="Pulse" name="pulse" hint="beats per minute">
        {({ id, describedBy, invalid }) => (
          <TextInput
            id={id}
            name="pulse"
            type="number"
            min={20}
            max={250}
            step={1}
            inputMode="numeric"
            aria-describedby={describedBy}
            invalid={invalid}
          />
        )}
      </Field>

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

      <SubmitButton className="w-full sm:w-auto">Save blood pressure</SubmitButton>
    </ActionForm>
  );
}
