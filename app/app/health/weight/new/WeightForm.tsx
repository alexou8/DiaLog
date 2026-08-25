'use client';

import { useActionState } from 'react';
import { useFormStatus } from 'react-dom';
import { addWeightAction, type RecordActionState } from '@/lib/actions/records';
import { toLocalInputValue } from '@/lib/domain/time';
import { Button } from '@/components/ui';
import { Field, FormStatus, Select, TextInput } from '@/components/ui/form';

function Submit() {
  const { pending } = useFormStatus();
  return (
    <Button type="submit" disabled={pending} className="w-full sm:w-auto">
      {pending ? 'Saving…' : 'Save weight'}
    </Button>
  );
}

export function WeightForm({ timezone }: { timezone: string }) {
  const [state, action] = useActionState<RecordActionState | null, FormData>(addWeightAction, null);
  const now = toLocalInputValue(new Date(), timezone);

  return (
    <form action={action} noValidate>
      <FormStatus
        status={state && !state.ok && state.message ? { ok: false, message: state.message } : null}
      />

      <div className="grid grid-cols-[1fr,auto] items-start gap-3">
        <Field label="Weight" required error={state?.errors?.weight} htmlFor="weight">
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
