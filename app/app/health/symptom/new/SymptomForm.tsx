'use client';

import { useActionState } from 'react';
import { useFormStatus } from 'react-dom';
import { addSymptomAction, type RecordActionState } from '@/lib/actions/records';
import { Button } from '@/components/ui';
import { Field, FormStatus, RadioCards, TextArea, TextInput } from '@/components/ui/form';

const COMMON = ['Tired', 'Thirsty', 'Headache', 'Dizzy', 'Shaky', 'Blurred vision', 'Nausea'];

function Submit() {
  const { pending } = useFormStatus();
  return (
    <Button type="submit" className="w-full text-lg" disabled={pending}>
      {pending ? 'Saving…' : 'Save'}
    </Button>
  );
}

export function SymptomForm({ defaultTime }: { defaultTime: string }) {
  const [state, action] = useActionState<RecordActionState | null, FormData>(
    addSymptomAction,
    null,
  );

  return (
    <form action={action} noValidate>
      <FormStatus
        status={state && !state.ok && state.message ? { ok: false, message: state.message } : null}
      />

      <Field
        label="What did you notice?"
        required
        hint="In your own words. The list offers a few common ones, but anything is fine."
        error={state?.errors?.symptom}
      >
        {({ id, describedBy, invalid }) => (
          <>
            <TextInput
              id={id}
              name="symptom"
              list="dl-common-symptoms"
              required
              autoFocus
              maxLength={120}
              aria-describedby={describedBy}
              invalid={invalid}
            />
            <datalist id="dl-common-symptoms">
              {COMMON.map((item) => (
                <option key={item} value={item} />
              ))}
            </datalist>
          </>
        )}
      </Field>

      <Field label="When?" required error={state?.errors?.takenAt}>
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
        name="severity"
        legend="How strong was it?"
        hint="Your own judgement. There is no right answer."
        columns={3}
        options={[
          { value: '1', label: 'Barely noticed' },
          { value: '2', label: 'Mild' },
          { value: '3', label: 'Moderate' },
          { value: '4', label: 'Strong' },
          { value: '5', label: 'Very strong' },
        ]}
      />

      <Field label="Anything else worth remembering?" error={state?.errors?.note}>
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
