'use client';

import { useActionState } from 'react';
import { useFormStatus } from 'react-dom';
import { completeOnboardingAction, skipOnboardingAction } from '@/lib/actions/onboarding';
import type { ActionState } from '@/lib/actions/auth';
import { Button, Card } from '@/components/ui';
import { Checkbox, Field, FormStatus, RadioCards, Select, TextInput } from '@/components/ui/form';
import { TIMEZONE_GROUPS } from '@/lib/timezones';

const GOALS = [
  { value: 'understand', label: 'Understand my patterns' },
  { value: 'steadier', label: 'Steadier readings day to day' },
  { value: 'food', label: 'See how food affects me' },
  { value: 'movement', label: 'Build a walking or exercise habit' },
  { value: 'appointments', label: 'Be better prepared for appointments' },
  { value: 'weight', label: 'Keep an eye on my weight' },
];

function Submit() {
  const { pending } = useFormStatus();
  return (
    <Button type="submit" className="w-full text-lg" disabled={pending}>
      {pending ? 'Saving…' : 'Finish setup'}
    </Button>
  );
}

export function OnboardingForm({
  defaultName,
  defaultTimezone,
}: {
  defaultName: string;
  defaultTimezone: string;
}) {
  const [state, action] = useActionState<ActionState | null, FormData>(completeOnboardingAction, null);

  // The browser knows the visitor's zone; offering it as the default saves a
  // scroll through a long list for almost everyone.
  const detected = typeof Intl !== 'undefined' ? Intl.DateTimeFormat().resolvedOptions().timeZone : defaultTimezone;

  return (
    <>
      <form action={action} className="mt-8 space-y-6" noValidate>
        <FormStatus status={state && !state.ok && state.message ? { ok: false, message: state.message } : null} />

        <Card>
          <Field label="What should we call you?" hint="Only used to greet you." error={state?.errors?.displayName}>
            {({ id, describedBy, invalid }) => (
              <TextInput
                id={id}
                name="displayName"
                defaultValue={defaultName}
                autoComplete="given-name"
                aria-describedby={describedBy}
                invalid={invalid}
              />
            )}
          </Field>
        </Card>

        <Card>
          <RadioCards
            name="glucoseUnit"
            legend="Which unit do your readings use?"
            hint="This is the number your meter shows. In Canada it is usually mmol/L."
            defaultValue="MMOLL"
            options={[
              { value: 'MMOLL', label: 'mmol/L', description: 'Readings look like 5.6 or 8.2' },
              { value: 'MGDL', label: 'mg/dL', description: 'Readings look like 101 or 148' },
            ]}
          />
        </Card>

        <Card>
          <RadioCards
            name="condition"
            legend="What brings you to DiaLog?"
            hint="This only affects the wording you see. It is never treated as a diagnosis, and you can choose not to say."
            columns={2}
            defaultValue="PREFER_NOT_TO_SAY"
            options={[
              { value: 'PREDIABETES', label: 'Prediabetes' },
              { value: 'TYPE_2', label: 'Type 2 diabetes' },
              { value: 'TYPE_1', label: 'Type 1 diabetes' },
              { value: 'GESTATIONAL', label: 'Gestational diabetes' },
              { value: 'CURIOUS', label: 'Curious about my glucose' },
              { value: 'PREFER_NOT_TO_SAY', label: 'Rather not say' },
            ]}
          />
        </Card>

        <Card>
          <Field label="Your time zone" hint="So that a reading at 8 in the morning is filed as the morning.">
            {({ id, describedBy }) => (
              <Select id={id} name="timezone" defaultValue={detected} aria-describedby={describedBy}>
                {TIMEZONE_GROUPS.map((group) => (
                  <optgroup key={group.label} label={group.label}>
                    {group.zones.map((zone) => (
                      <option key={zone.id} value={zone.id}>
                        {zone.label}
                      </option>
                    ))}
                  </optgroup>
                ))}
              </Select>
            )}
          </Field>
        </Card>

        <Card>
          <fieldset>
            <legend className="mb-1.5 text-base font-semibold">
              What would you like to get out of this?
            </legend>
            <p className="mb-3 text-sm text-ink-muted">
              Pick any that apply, or none. This shapes which observations get shown first.
            </p>
            {GOALS.map((goal) => (
              <Checkbox key={goal.value} name="goals" label={goal.label} value={goal.value} />
            ))}
          </fieldset>
        </Card>

        <Submit />
      </form>

      <form action={skipOnboardingAction} className="mt-4 text-center">
        <button type="submit" className="dl-target text-ink-muted underline underline-offset-4">
          Skip for now
        </button>
      </form>
    </>
  );
}
