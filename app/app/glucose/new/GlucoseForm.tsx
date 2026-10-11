'use client';

import { useActionState } from 'react';
import type { GlucoseUnit } from '@prisma/client';
import { addGlucoseAction, type RecordActionState } from '@/lib/actions/records';
import {
  ActionForm,
  Field,
  RadioCards,
  SubmitButton,
  TextArea,
  TextInput,
} from '@/components/ui/form';

const CONTEXTS = [
  { value: 'FASTING', label: 'Fasting', description: 'First thing, before eating' },
  { value: 'BEFORE_MEAL', label: 'Before a meal' },
  { value: 'AFTER_MEAL', label: 'After a meal', description: 'Usually one to two hours after' },
  { value: 'BEDTIME', label: 'At bedtime' },
  { value: 'RANDOM', label: 'Any other time' },
] as const;

export function GlucoseForm({
  unit,
  unitLabel,
  defaultTime,
  targetHint,
}: {
  unit: GlucoseUnit;
  unitLabel: string;
  defaultTime: string;
  targetHint: string;
}) {
  const [state, action, pending] = useActionState<RecordActionState | null, FormData>(
    addGlucoseAction,
    null,
  );
  const readingLabel = `Your reading (${unitLabel})`;

  return (
    <ActionForm
      action={action}
      state={state}
      pending={pending}
      labels={{
        value: readingLabel,
        takenAt: 'When was it taken?',
        context: 'What was happening around then?',
        note: 'Note',
      }}
    >
      <input type="hidden" name="unit" value={unit} />

      <Field label={readingLabel} required hint={targetHint} name="value">
        {({ id, describedBy, invalid }) => (
          <TextInput
            id={id}
            name="value"
            type="number"
            step={unit === 'MMOLL' ? '0.1' : '1'}
            inputMode="decimal"
            required
            // Only on a fresh form. After a failed save, focus belongs to the
            // error summary, not back on the first field.
            autoFocus={!state}
            className="dl-numeric text-2xl"
            aria-describedby={describedBy}
            invalid={invalid}
          />
        )}
      </Field>

      <Field label="When was it taken?" required name="takenAt">
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

      <RadioCards
        name="context"
        legend="What was happening around then?"
        hint="This helps DiaLog compare like with like. Choose 'Any other time' if you are not sure."
        defaultValue="RANDOM"
        columns={2}
        options={[...CONTEXTS]}
      />

      <Field
        label="A note, if you want one"
        hint="For example: “felt shaky”, or “after a long walk”."
        name="note"
      >
        {({ id, describedBy, invalid }) => (
          <TextArea
            id={id}
            name="note"
            rows={2}
            maxLength={500}
            aria-describedby={describedBy}
            invalid={invalid}
          />
        )}
      </Field>

      <SubmitButton className="w-full text-lg">Save reading</SubmitButton>
    </ActionForm>
  );
}
