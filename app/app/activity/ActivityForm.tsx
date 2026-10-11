'use client';

import { useActionState } from 'react';
import { addExerciseAction, type RecordActionState } from '@/lib/actions/records';
import { toLocalInputValue } from '@/lib/domain/time';
import { ActionForm, Field, RadioCards, SubmitButton, TextInput } from '@/components/ui/form';

const COMMON_ACTIVITIES = [
  'Walking',
  'Running',
  'Cycling',
  'Strength training',
  'Swimming',
  'Sports',
  'Housework/gardening',
];

const INTENSITY_OPTIONS: {
  value: 'LIGHT' | 'MODERATE' | 'VIGOROUS';
  label: string;
  description: string;
}[] = [
  { value: 'LIGHT', label: 'Light', description: 'You could sing' },
  { value: 'MODERATE', label: 'Moderate', description: 'You could talk but not sing' },
  { value: 'VIGOROUS', label: 'Vigorous', description: 'Talking is hard' },
];

export function ActivityForm({ timezone }: { timezone: string }) {
  const [state, action, pending] = useActionState<RecordActionState | null, FormData>(
    addExerciseAction,
    null,
  );
  const defaultTakenAt = toLocalInputValue(new Date(), timezone);

  return (
    <ActionForm
      action={action}
      state={state}
      pending={pending}
      labels={{
        activity: 'Activity',
        takenAt: 'When did it start',
        durationMin: 'Duration',
        distanceKm: 'Distance',
        steps: 'Steps',
        note: 'Note',
        intensity: 'Intensity',
      }}
    >
      <Field label="Activity" required name="activity">
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

      <Field label="When did it start" required name="takenAt">
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

      <Field label="Duration" required name="durationMin" hint="minutes">
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
        <Field label="Distance" name="distanceKm" hint="kilometres">
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
        <Field label="Steps" name="steps">
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

      <SubmitButton className="w-full sm:w-auto">Save activity</SubmitButton>
    </ActionForm>
  );
}
