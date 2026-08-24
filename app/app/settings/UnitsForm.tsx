'use client';

import { useActionState, useState } from 'react';
import { useFormStatus } from 'react-dom';
import type { Profile } from '@prisma/client';
import { updatePreferencesAction, type ActionState } from '@/lib/actions/preferences';
import { fromMgdl, toMgdl, unitLabel, unitPrecision } from '@/lib/domain/units';
import { Button, Callout } from '@/components/ui';
import { Field, FormStatus, RadioCards, TextInput } from '@/components/ui/form';
import { HiddenBool } from './HiddenBool';

type Unit = Profile['glucoseUnit'];

function formatForUnit(mgdl: number, unit: Unit): string {
  return fromMgdl(mgdl, unit).toFixed(unitPrecision(unit));
}

function Submit() {
  const { pending } = useFormStatus();
  return (
    <Button type="submit" disabled={pending}>
      {pending ? 'Saving…' : 'Save units and targets'}
    </Button>
  );
}

export function UnitsForm({ profile }: { profile: Profile }) {
  const [state, action] = useActionState<ActionState | null, FormData>(updatePreferencesAction, null);
  const [unit, setUnit] = useState<Unit>(profile.glucoseUnit);
  const [low, setLow] = useState(() => formatForUnit(profile.targetLowMgdl, profile.glucoseUnit));
  const [high, setHigh] = useState(() => formatForUnit(profile.targetHighMgdl, profile.glucoseUnit));

  function handleUnitChange(event: React.ChangeEvent<HTMLDivElement>) {
    const target = event.target as HTMLInputElement;
    if (target.name !== 'glucoseUnit') return;
    const nextUnit = target.value as Unit;
    if (nextUnit === unit) return;
    const lowMgdl = toMgdl(Number.parseFloat(low) || 0, unit);
    const highMgdl = toMgdl(Number.parseFloat(high) || 0, unit);
    setLow(formatForUnit(lowMgdl, nextUnit));
    setHigh(formatForUnit(highMgdl, nextUnit));
    setUnit(nextUnit);
  }

  const step = unitPrecision(unit) === 0 ? '1' : '0.1';

  return (
    <form action={action} noValidate className="rounded-[var(--radius-card)] border border-line bg-surface p-5 sm:p-6">
      <FormStatus status={state && state.message ? { ok: state.ok, message: state.message } : null} />

      <div onChange={handleUnitChange}>
        <RadioCards
          name="glucoseUnit"
          legend="How glucose numbers are shown"
          hint="This changes how readings are displayed. It never changes the numbers stored in your account."
          defaultValue={profile.glucoseUnit}
          options={[
            { value: 'MMOLL', label: 'mmol/L', description: 'Common across Canada' },
            { value: 'MGDL', label: 'mg/dL', description: 'Common in the United States' },
          ]}
        />
      </div>

      <fieldset className="mb-2">
        <legend className="mb-1.5 text-base font-semibold">Your personal target range</legend>
        <p className="mb-3 text-sm text-ink-muted">
          This is the range you and your healthcare professional consider reasonable for you — not a
          clinical threshold, and not medical advice. DiaLog uses it only to describe your own readings
          (for example, &ldquo;in your target range&rdquo;); it never changes what is stored.
        </p>
        <div className="grid gap-4 sm:grid-cols-2">
          <Field label={`Low end (${unitLabel(unit)})`} required error={state?.errors?.targetLow}>
            {({ id, describedBy, invalid }) => (
              <TextInput
                id={id}
                name="targetLow"
                type="number"
                inputMode="decimal"
                step={step}
                required
                value={low}
                onChange={(e) => setLow(e.target.value)}
                aria-describedby={describedBy}
                invalid={invalid}
              />
            )}
          </Field>
          <Field label={`High end (${unitLabel(unit)})`} required error={state?.errors?.targetHigh}>
            {({ id, describedBy, invalid }) => (
              <TextInput
                id={id}
                name="targetHigh"
                type="number"
                inputMode="decimal"
                step={step}
                required
                value={high}
                onChange={(e) => setHigh(e.target.value)}
                aria-describedby={describedBy}
                invalid={invalid}
              />
            )}
          </Field>
        </div>
      </fieldset>

      <Callout tone="info" icon="ℹ️">
        Changing your target range changes how readings are labelled going forward. It does not
        re-interpret past observations or alter any reading you already logged.
      </Callout>

      {/* Fields owned by other sections of this page, carried through unchanged. */}
      <input type="hidden" name="displayName" value={profile.displayName ?? ''} />
      <input type="hidden" name="locale" value={profile.locale} />
      <input type="hidden" name="timezone" value={profile.timezone} />
      <input type="hidden" name="detailLevel" value={profile.detailLevel} />
      <HiddenBool name="largeText" value={profile.largeText} />
      <HiddenBool name="reduceMotion" value={profile.reduceMotion} />
      <HiddenBool name="aiEnabled" value={profile.aiEnabled} />
      <HiddenBool name="externalAiConsent" value={profile.externalAiConsentAt != null} />

      <div className="mt-5">
        <Submit />
      </div>
    </form>
  );
}
