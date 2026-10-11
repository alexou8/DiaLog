'use client';

import { useActionState } from 'react';
import { addSymptomAction, type RecordActionState } from '@/lib/actions/records';
import {
  ActionForm,
  Field,
  RadioCards,
  SubmitButton,
  TextArea,
  TextInput,
} from '@/components/ui/form';

const COMMON = ['Tired', 'Thirsty', 'Headache', 'Dizzy', 'Shaky', 'Blurred vision', 'Nausea'];

export function SymptomForm({ defaultTime }: { defaultTime: string }) {
  const [state, action, pending] = useActionState<RecordActionState | null, FormData>(
    addSymptomAction,
    null,
  );

  return (
    <ActionForm
      action={action}
      state={state}
      pending={pending}
      labels={{
        symptom: 'What did you notice?',
        takenAt: 'When?',
        note: 'Anything else worth remembering?',
        severity: 'How strong was it?',
      }}
    >
      <Field
        label="What did you notice?"
        required
        hint="In your own words. The list offers a few common ones, but anything is fine."
        name="symptom"
      >
        {({ id, describedBy, invalid }) => (
          <>
            <TextInput
              id={id}
              name="symptom"
              list="dl-common-symptoms"
              required
              autoFocus={!state}
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

      <Field label="When?" required name="takenAt">
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

      <Field label="Anything else worth remembering?" name="note">
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

      <SubmitButton className="w-full text-lg">Save</SubmitButton>
    </ActionForm>
  );
}
