'use client';

import { useActionState, useEffect, useState } from 'react';
import { completeOnboardingAction, skipOnboardingAction } from '@/lib/actions/onboarding';
import type { FormActionState } from '@/components/ui/form';
import { Card } from '@/components/ui';
import {
  ActionForm,
  Checkbox,
  Field,
  RadioCards,
  Select,
  SubmitButton,
  TextInput,
} from '@/components/ui/form';
import { ALL_TIMEZONE_IDS, TIMEZONE_GROUPS } from '@/lib/timezones';

const GOALS = [
  { value: 'understand', label: 'Understand my patterns' },
  { value: 'steadier', label: 'Steadier readings day to day' },
  { value: 'food', label: 'See how food affects me' },
  { value: 'movement', label: 'Build a walking or exercise habit' },
  { value: 'appointments', label: 'Be better prepared for appointments' },
  { value: 'weight', label: 'Keep an eye on my weight' },
];

/** The zone this device reports, or null where Intl cannot say. */
function browserTimeZone(): string | null {
  try {
    return Intl.DateTimeFormat().resolvedOptions().timeZone || null;
  } catch {
    return null;
  }
}

export function OnboardingForm({
  defaultName,
  defaultTimezone,
}: {
  defaultName: string;
  defaultTimezone: string;
}) {
  const [state, action, pending] = useActionState<FormActionState | null, FormData>(
    completeOnboardingAction,
    null,
  );

  // The server renders the profile's zone and the client's first render uses
  // the same value, so hydration agrees. Reading the browser's zone during
  // render used to run on the server too, where it is the *server's* zone
  // (UTC on Vercel): a submit before hydration posted that, and every reading
  // was then filed under the wrong day and time of day. The browser's zone is
  // applied only after mount, here, and said out loud in the hint.
  const submitted = state?.values?.timezone;
  const [timezone, setTimezone] = useState(
    typeof submitted === 'string' ? submitted : defaultTimezone,
  );
  const [detected, setDetected] = useState<string | null>(null);
  // A zone the person already submitted (a no-JavaScript round trip) is
  // their choice and outranks what the device reports.
  const [hadSubmission] = useState(typeof submitted === 'string');

  useEffect(() => {
    const zone = browserTimeZone();
    if (!zone) return;
    setDetected(zone);
    if (!hadSubmission) setTimezone(zone);
  }, [hadSubmission]);

  // A device zone outside the curated list is offered as its own option
  // rather than letting the <select> silently fall back to its first entry.
  const extraZone = [timezone, detected].find(
    (zone): zone is string => !!zone && !ALL_TIMEZONE_IDS.includes(zone),
  );

  return (
    <>
      <ActionForm
        action={action}
        state={state}
        pending={pending}
        className="mt-8 flex flex-col gap-6"
        labels={{
          displayName: 'What should we call you?',
          glucoseUnit: 'Which unit do your readings use?',
          condition: 'What brings you to DiaLog?',
          timezone: 'Your time zone',
          goals: 'What would you like to get out of this?',
        }}
      >
        <Card>
          <Field label="What should we call you?" hint="Only used to greet you." name="displayName">
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
          <Field
            label="Your time zone"
            hint={
              detected
                ? `So that a reading at 8 in the morning is filed as the morning. This device reports ${detected}; change it if that is not where you live.`
                : 'So that a reading at 8 in the morning is filed as the morning.'
            }
            name="timezone"
          >
            {({ id, describedBy, invalid }) => (
              <Select
                id={id}
                name="timezone"
                value={timezone}
                onChange={(event) => setTimezone(event.target.value)}
                aria-describedby={describedBy}
                invalid={invalid}
              >
                {extraZone ? (
                  <optgroup label="This device">
                    <option value={extraZone}>{extraZone.replace(/_/g, ' ')}</option>
                  </optgroup>
                ) : null}
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

        <SubmitButton className="w-full text-lg">Finish setup</SubmitButton>
      </ActionForm>

      <form action={skipOnboardingAction} className="mt-4 text-center">
        <button type="submit" className="dl-target text-ink-muted underline underline-offset-4">
          Skip for now
        </button>
      </form>
    </>
  );
}
