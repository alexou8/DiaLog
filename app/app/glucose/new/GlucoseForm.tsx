'use client';

import { useActionState } from 'react';
import { useFormStatus } from 'react-dom';
import type { GlucoseUnit } from '@prisma/client';
import { addGlucoseAction, type RecordActionState } from '@/lib/actions/records';
import { Button } from '@/components/ui';
import { Field, FormStatus, RadioCards, TextArea, TextInput } from '@/components/ui/form';

const CONTEXTS = [
  { value: 'FASTING', label: 'Fasting', description: 'First thing, before eating' },
  { value: 'BEFORE_MEAL', label: 'Before a meal' },
  { value: 'AFTER_MEAL', label: 'After a meal', description: 'Usually one to two hours after' },
  { value: 'BEDTIME', label: 'At bedtime' },
  { value: 'RANDOM', label: 'Any other time' },
] as const;

function Submit() {
  const { pending } = useFormStatus();
  return (
    <Button type="submit" className="w-full text-lg" disabled={pending}>
      {pending ? 'Saving…' : 'Save reading'}
    </Button>
  );
}

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
  const [state, action] = useActionState<RecordActionState | null, FormData>(
    addGlucoseAction,
    null,
  );

  return (
    <form action={action} noValidate>
      <FormStatus
        status={state && !state.ok && state.message ? { ok: false, message: state.message } : null}
      />
      <input type="hidden" name="unit" value={unit} />

      <Field
        label={`Your reading (${unitLabel})`}
        required
        hint={targetHint}
        error={state?.errors?.value}
      >
        {({ id, describedBy, invalid }) => (
          <TextInput
            id={id}
            name="value"
            type="number"
            step={unit === 'MMOLL' ? '0.1' : '1'}
            inputMode="decimal"
            required
            autoFocus
            className="text-2xl"
            aria-describedby={describedBy}
            invalid={invalid}
          />
        )}
      </Field>

      <Field label="When was it taken?" required error={state?.errors?.takenAt}>
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
        error={state?.errors?.note}
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

      <Submit />
    </form>
  );
}
