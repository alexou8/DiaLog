'use client';

import { useActionState } from 'react';
import { useFormStatus } from 'react-dom';
import { addBloodPressureAction, type RecordActionState } from '@/lib/actions/records';
import { toLocalInputValue } from '@/lib/domain/time';
import { Button } from '@/components/ui';
import { Field, FormStatus, TextInput } from '@/components/ui/form';

function Submit() {
  const { pending } = useFormStatus();
  return (
    <Button type="submit" disabled={pending} className="w-full sm:w-auto">
      {pending ? 'Saving…' : 'Save blood pressure'}
    </Button>
  );
}

export function BloodPressureForm({ timezone }: { timezone: string }) {
  const [state, action] = useActionState<RecordActionState | null, FormData>(
    addBloodPressureAction,
    null,
  );
  const now = toLocalInputValue(new Date(), timezone);

  return (
    <form action={action} noValidate>
      <FormStatus
        status={state && !state.ok && state.message ? { ok: false, message: state.message } : null}
      />

      <div className="grid gap-4 sm:grid-cols-2">
        <Field
          label="Upper number (systolic)"
          required
          error={state?.errors?.systolic}
          htmlFor="systolic"
        >
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
        <Field
          label="Lower number (diastolic)"
          required
          error={state?.errors?.diastolic}
          htmlFor="diastolic"
        >
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

      <Field label="Pulse" error={state?.errors?.pulse} htmlFor="pulse" hint="beats per minute">
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
