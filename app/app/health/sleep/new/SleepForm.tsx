'use client';

import { useActionState } from 'react';
import { addSleepAction, type RecordActionState } from '@/lib/actions/records';
import { toLocalInputValue } from '@/lib/domain/time';
import { ActionForm, Field, RadioCards, SubmitButton, TextInput } from '@/components/ui/form';

const QUALITY_OPTIONS: {
  value: '1' | '2' | '3' | '4' | '5';
  label: string;
  description: string;
}[] = [
  { value: '1', label: '1: Poor', description: 'Woke up feeling exhausted' },
  { value: '2', label: '2: Fair', description: 'Restless, several wake-ups' },
  { value: '3', label: '3: Okay', description: 'An average night' },
  { value: '4', label: '4: Good', description: 'Slept well, woke rested' },
  { value: '5', label: '5: Excellent', description: 'Deep, uninterrupted sleep' },
];

export function SleepForm({ timezone }: { timezone: string }) {
  const [state, action, pending] = useActionState<RecordActionState | null, FormData>(
    addSleepAction,
    null,
  );
  const now = toLocalInputValue(new Date(), timezone);

  return (
    <ActionForm
      action={action}
      state={state}
      pending={pending}
      labels={{
        takenAt: 'Bedtime',
        endedAt: 'Wake time',
        note: 'Note',
        quality: 'Sleep quality',
      }}
    >
      <Field label="Bedtime" required name="takenAt">
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

      <Field label="Wake time" required name="endedAt">
        {({ id, describedBy, invalid }) => (
          <TextInput
            id={id}
            name="endedAt"
            type="datetime-local"
            required
            defaultValue={now}
            aria-describedby={describedBy}
            invalid={invalid}
          />
        )}
      </Field>

      <RadioCards
        name="quality"
        legend="Sleep quality"
        hint="How rested did you feel?"
        options={QUALITY_OPTIONS}
        columns={1}
      />

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

      <SubmitButton className="w-full sm:w-auto">Save sleep</SubmitButton>
    </ActionForm>
  );
}
