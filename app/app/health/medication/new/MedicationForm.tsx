'use client';

import { useActionState } from 'react';
import { useFormStatus } from 'react-dom';
import { addMedicationAction, type RecordActionState } from '@/lib/actions/records';
import { toLocalInputValue } from '@/lib/domain/time';
import { Button, Callout } from '@/components/ui';
import { Field, FormStatus, TextInput } from '@/components/ui/form';

function Submit() {
  const { pending } = useFormStatus();
  return (
    <Button type="submit" disabled={pending} className="w-full sm:w-auto">
      {pending ? 'Saving…' : 'Save medication event'}
    </Button>
  );
}

export function MedicationForm({ timezone }: { timezone: string }) {
  const [state, action] = useActionState<RecordActionState | null, FormData>(addMedicationAction, null);
  const now = toLocalInputValue(new Date(), timezone);

  return (
    <form action={action} noValidate>
      <FormStatus status={state && !state.ok && state.message ? { ok: false, message: state.message } : null} />

      <Callout tone="info" icon="ℹ️">
        This just records that you took something, and when — so you can see it next to your
        readings. DiaLog never calculates or suggests a dose.
      </Callout>

      <div className="mt-5">
        <Field label="Medication name" required error={state?.errors?.name} htmlFor="name" hint="As it appears on the package">
          {({ id, describedBy, invalid }) => (
            <TextInput id={id} name="name" required maxLength={120} aria-describedby={describedBy} invalid={invalid} />
          )}
        </Field>
      </div>

      <Field label="When" required error={state?.errors?.takenAt} htmlFor="takenAt">
        {({ id, describedBy, invalid }) => (
          <TextInput id={id} name="takenAt" type="datetime-local" required defaultValue={now} aria-describedby={describedBy} invalid={invalid} />
        )}
      </Field>

      <Field label="Dose" error={state?.errors?.dose} htmlFor="dose" hint="As you took it, e.g. “10 mg” or “1 tablet”. Recorded exactly as typed.">
        {({ id, describedBy, invalid }) => (
          <TextInput id={id} name="dose" maxLength={60} aria-describedby={describedBy} invalid={invalid} />
        )}
      </Field>

      <Field label="Route" error={state?.errors?.route} htmlFor="route" hint="e.g. oral, injection">
        {({ id, describedBy, invalid }) => (
          <TextInput id={id} name="route" maxLength={40} aria-describedby={describedBy} invalid={invalid} />
        )}
      </Field>

      <Field label="Note" error={state?.errors?.note} htmlFor="note">
        {({ id, describedBy, invalid }) => (
          <TextInput id={id} name="note" maxLength={500} aria-describedby={describedBy} invalid={invalid} />
        )}
      </Field>

      <Submit />
    </form>
  );
}
