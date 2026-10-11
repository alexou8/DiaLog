'use client';

import { useActionState } from 'react';
import type { Profile } from '@prisma/client';
import { updatePreferencesAction, type ActionState } from '@/lib/actions/preferences';
import { fromMgdl } from '@/lib/domain/units';
import { LOCALE_LABELS, LOCALES } from '@/lib/i18n/dictionaries';
import { ActionForm, Field, Select, SubmitButton } from '@/components/ui/form';
import { HiddenBool } from './HiddenBool';
import { TIMEZONE_OPTIONS } from './timezones';

export function RegionForm({ profile }: { profile: Profile }) {
  const [state, action, pending] = useActionState<ActionState | null, FormData>(
    updatePreferencesAction,
    null,
  );

  return (
    <ActionForm
      action={action}
      state={state}
      pending={pending}
      className="rounded-[var(--radius-card)] border border-line bg-surface p-5 sm:p-6"
      labels={{
        locale: 'Language',
        timezone: 'Time zone',
      }}
    >
      <div className="grid gap-4 sm:grid-cols-2">
        <Field
          label="Language"
          required
          name="locale"
          hint="French is a partial translation today. Some screens still show in English."
        >
          {({ id, describedBy, invalid }) => (
            <Select
              id={id}
              name="locale"
              required
              defaultValue={profile.locale}
              aria-describedby={describedBy}
              invalid={invalid}
            >
              {LOCALES.map((code) => (
                <option key={code} value={code}>
                  {LOCALE_LABELS[code]}
                </option>
              ))}
            </Select>
          )}
        </Field>

        <Field
          label="Time zone"
          required
          name="timezone"
          hint="Used to group your readings by day and to time-stamp new entries."
        >
          {({ id, describedBy, invalid }) => (
            <Select
              id={id}
              name="timezone"
              required
              defaultValue={profile.timezone}
              aria-describedby={describedBy}
              invalid={invalid}
            >
              {TIMEZONE_OPTIONS.map((tz) => (
                <option key={tz.value} value={tz.value}>
                  {tz.label}
                </option>
              ))}
            </Select>
          )}
        </Field>
      </div>

      <input type="hidden" name="displayName" value={profile.displayName ?? ''} />
      <input type="hidden" name="glucoseUnit" value={profile.glucoseUnit} />
      <input
        type="hidden"
        name="targetLow"
        value={fromMgdl(profile.targetLowMgdl, profile.glucoseUnit)}
      />
      <input
        type="hidden"
        name="targetHigh"
        value={fromMgdl(profile.targetHighMgdl, profile.glucoseUnit)}
      />
      <input type="hidden" name="detailLevel" value={profile.detailLevel} />
      <HiddenBool name="largeText" value={profile.largeText} />
      <HiddenBool name="reduceMotion" value={profile.reduceMotion} />
      <HiddenBool name="aiEnabled" value={profile.aiEnabled} />
      <HiddenBool name="externalAiConsent" value={profile.externalAiConsentAt != null} />

      <div className="mt-5">
        <SubmitButton>Save language and region</SubmitButton>
      </div>
    </ActionForm>
  );
}
