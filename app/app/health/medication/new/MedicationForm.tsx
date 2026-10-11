'use client';

import { useActionState } from 'react';
import { addMedicationAction, type RecordActionState } from '@/lib/actions/records';
import { toLocalInputValue } from '@/lib/domain/time';
import { Callout } from '@/components/ui';
import { ActionForm, Field, SubmitButton, TextInput } from '@/components/ui/form';

export function MedicationForm({ timezone }: { timezone: string }) {
  const [state, action, pending] = useActionState<RecordActionState | null, FormData>(
    addMedicationAction,
    null,
  );
  const now = toLocalInputValue(new Date(), timezone);

  return (
    <ActionForm
      action={action}
      state={state}
      pending={pending}
      labels={{
        name: 'Medication name',
        takenAt: 'When',
        dose: 'Dose',
        route: 'Route',
        note: 'Note',
      }}
    >
      <Callout tone="info" icon="info">
        This just records that you took something, and when, so you can see it next to your
        readings. DiaLog never calculates or suggests a dose.
      </Callout>

      <div className="mt-5">
        <Field label="Medication name" required name="name" hint="As it appears on the package">
          {({ id, describedBy, invalid }) => (
            <TextInput
              id={id}
              name="name"
              required
              maxLength={120}
              aria-describedby={describedBy}
              invalid={invalid}
            />
          )}
        </Field>
      </div>

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

      <Field
        label="Dose"
        name="dose"
        hint="As you took it, e.g. “10 mg” or “1 tablet”. Recorded exactly as typed."
      >
        {({ id, describedBy, invalid }) => (
          <TextInput
            id={id}
            name="dose"
            maxLength={60}
            aria-describedby={describedBy}
            invalid={invalid}
          />
        )}
      </Field>

      <Field label="Route" name="route" hint="e.g. oral, injection">
        {({ id, describedBy, invalid }) => (
          <TextInput
            id={id}
            name="route"
            maxLength={40}
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

      <SubmitButton className="w-full sm:w-auto">Save medication event</SubmitButton>
    </ActionForm>
  );
}
