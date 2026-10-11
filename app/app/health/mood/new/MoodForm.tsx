'use client';

import { useActionState } from 'react';
import { addMoodAction, type RecordActionState } from '@/lib/actions/records';
import { toLocalInputValue } from '@/lib/domain/time';
import { ActionForm, Field, RadioCards, SubmitButton, TextInput } from '@/components/ui/form';

const MOOD_OPTIONS: { value: '1' | '2' | '3' | '4' | '5'; label: string }[] = [
  { value: '1', label: 'Very low' },
  { value: '2', label: 'Low' },
  { value: '3', label: 'Okay' },
  { value: '4', label: 'Good' },
  { value: '5', label: 'Great' },
];

const STRESS_OPTIONS: { value: '1' | '2' | '3' | '4' | '5'; label: string }[] = [
  { value: '1', label: 'Calm' },
  { value: '2', label: 'Mild' },
  { value: '3', label: 'Moderate' },
  { value: '4', label: 'High' },
  { value: '5', label: 'Very stressed' },
];

export function MoodForm({ timezone }: { timezone: string }) {
  const [state, action, pending] = useActionState<RecordActionState | null, FormData>(
    addMoodAction,
    null,
  );
  const now = toLocalInputValue(new Date(), timezone);

  return (
    <ActionForm
      action={action}
      state={state}
      pending={pending}
      labels={{
        takenAt: 'When',
        note: 'Note',
        mood: 'How are you feeling?',
        stress: 'Stress level',
      }}
    >
      <RadioCards name="mood" legend="How are you feeling?" options={MOOD_OPTIONS} columns={1} />
      <RadioCards name="stress" legend="Stress level" options={STRESS_OPTIONS} columns={1} />

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

      <SubmitButton className="w-full sm:w-auto">Save mood</SubmitButton>
    </ActionForm>
  );
}
