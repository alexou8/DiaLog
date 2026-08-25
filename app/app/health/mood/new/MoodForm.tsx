'use client';

import { useActionState } from 'react';
import { useFormStatus } from 'react-dom';
import { addMoodAction, type RecordActionState } from '@/lib/actions/records';
import { toLocalInputValue } from '@/lib/domain/time';
import { Button } from '@/components/ui';
import { Field, FormStatus, RadioCards, TextInput } from '@/components/ui/form';

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

function Submit() {
  const { pending } = useFormStatus();
  return (
    <Button type="submit" disabled={pending} className="w-full sm:w-auto">
      {pending ? 'Saving…' : 'Save mood'}
    </Button>
  );
}

export function MoodForm({ timezone }: { timezone: string }) {
  const [state, action] = useActionState<RecordActionState | null, FormData>(addMoodAction, null);
  const now = toLocalInputValue(new Date(), timezone);

  return (
    <form action={action} noValidate>
      <FormStatus
        status={state && !state.ok && state.message ? { ok: false, message: state.message } : null}
      />

      <RadioCards name="mood" legend="How are you feeling?" options={MOOD_OPTIONS} columns={1} />
      <RadioCards name="stress" legend="Stress level" options={STRESS_OPTIONS} columns={1} />

      <Field label="When" required error={state?.errors?.takenAt} htmlFor="takenAt">
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

      <Field label="Note" error={state?.errors?.note} htmlFor="note">
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

      <Submit />
    </form>
  );
}
