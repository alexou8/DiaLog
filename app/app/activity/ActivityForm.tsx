'use client';

import { useActionState } from 'react';
import { useFormStatus } from 'react-dom';
import { addExerciseAction, type RecordActionState } from '@/lib/actions/records';
import { toLocalInputValue } from '@/lib/domain/time';
import { Button } from '@/components/ui';
import { Field, FormStatus, RadioCards, TextInput } from '@/components/ui/form';

const COMMON_ACTIVITIES = [
  'Walking',
  'Running',
  'Cycling',
  'Strength training',
  'Swimming',
  'Sports',
  'Housework/gardening',
];

const INTENSITY_OPTIONS: { value: 'LIGHT' | 'MODERATE' | 'VIGOROUS'; label: string; description: string }[] = [
  { value: 'LIGHT', label: 'Light', description: 'You could sing' },
  { value: 'MODERATE', label: 'Moderate', description: 'You could talk but not sing' },
  { value: 'VIGOROUS', label: 'Vigorous', description: 'Talking is hard' },
];

function Submit() {
  const { pending } = useFormStatus();
  return (
    <Button type="submit" disabled={pending} className="w-full sm:w-auto">
      {pending ? 'Saving…' : 'Save activity'}
    </Button>
  );
}

export function ActivityForm({ timezone }: { timezone: string }) {
  const [state, action] = useActionState<RecordActionState | null, FormData>(addExerciseAction, null);
  const defaultTakenAt = toLocalInputValue(new Date(), timezone);

  return (
    <form action={action} noValidate>
      <FormStatus status={state && !state.ok && state.message ? { ok: false, message: state.message } : null} />

      <Field label="Activity" required error={state?.errors?.activity} htmlFor="activity">
        {({ id, describedBy, invalid }) => (
          <>
            <TextInput
              id={id}
              name="activity"
              required
              maxLength={80}
              list="activity-options"
              placeholder="e.g. Walking"
              aria-describedby={describedBy}
              invalid={invalid}
            />
            <datalist id="activity-options">
              {COMMON_ACTIVITIES.map((a) => (
                <option key={a} value={a} />
              ))}
              <option value="Other" />
            </datalist>
          </>
        )}
      </Field>

      <Field label="When did it start" required error={state?.errors?.takenAt} htmlFor="takenAt">
        {({ id, describedBy, invalid }) => (
          <TextInput
            id={id}
            name="takenAt"
            type="datetime-local"
            required
            defaultValue={defaultTakenAt}
            aria-describedby={describedBy}
            invalid={invalid}
          />
        )}
      </Field>

      <Field label="Duration" required error={state?.errors?.durationMin} htmlFor="durationMin" hint="minutes">
        {({ id, describedBy, invalid }) => (
          <TextInput
            id={id}
            name="durationMin"
            type="number"
            required
            min={1}
            max={1440}
            step={1}
            inputMode="numeric"
            aria-describedby={describedBy}
            invalid={invalid}
          />
        )}
      </Field>

      <RadioCards
        name="intensity"
        legend="Intensity"
        options={INTENSITY_OPTIONS}
        defaultValue="MODERATE"
        columns={1}
      />

      <div className="grid gap-4 sm:grid-cols-2">
        <Field label="Distance" error={state?.errors?.distanceKm} htmlFor="distanceKm" hint="kilometres">
          {({ id, describedBy, invalid }) => (
            <TextInput
              id={id}
              name="distanceKm"
              type="number"
              min={0}
              step="any"
              inputMode="decimal"
              aria-describedby={describedBy}
              invalid={invalid}
            />
          )}
        </Field>
        <Field label="Steps" error={state?.errors?.steps} htmlFor="steps">
          {({ id, describedBy, invalid }) => (
            <TextInput
              id={id}
              name="steps"
              type="number"
              min={0}
              step={1}
              inputMode="numeric"
              aria-describedby={describedBy}
              invalid={invalid}
            />
          )}
        </Field>
      </div>

      <Field label="Note" error={state?.errors?.note} htmlFor="note">
        {({ id, describedBy, invalid }) => (
          <TextInput id={id} name="note" maxLength={500} aria-describedby={describedBy} invalid={invalid} />
        )}
      </Field>

      <Submit />
    </form>
  );
}
