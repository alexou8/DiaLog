'use client';

import { useActionState } from 'react';
import { useFormStatus } from 'react-dom';
import type { Profile } from '@prisma/client';
import { updatePreferencesAction, type ActionState } from '@/lib/actions/preferences';
import { fromMgdl } from '@/lib/domain/units';
import { Button } from '@/components/ui';
import { Checkbox, FormStatus, RadioCards } from '@/components/ui/form';
import { HiddenBool } from './HiddenBool';

function Submit() {
  const { pending } = useFormStatus();
  return (
    <Button type="submit" disabled={pending}>
      {pending ? 'Saving…' : 'Save display settings'}
    </Button>
  );
}

export function DisplayForm({ profile }: { profile: Profile }) {
  const [state, action] = useActionState<ActionState | null, FormData>(
    updatePreferencesAction,
    null,
  );

  return (
    <form
      action={action}
      noValidate
      className="rounded-[var(--radius-card)] border border-line bg-surface p-5 sm:p-6"
    >
      <FormStatus
        status={state && state.message ? { ok: state.ok, message: state.message } : null}
      />

      <Checkbox
        name="largeText"
        label="Larger text"
        description="Increases text size across DiaLog for easier reading."
        defaultChecked={profile.largeText}
      />
      <Checkbox
        name="reduceMotion"
        label="Reduce motion"
        description="Turns off animated transitions and chart motion."
        defaultChecked={profile.reduceMotion}
      />

      <RadioCards
        name="detailLevel"
        legend="How much detail to show on your dashboard"
        hint="You can change this at any time — it only affects how much is shown, not what is recorded."
        defaultValue={profile.detailLevel}
        columns={3}
        options={[
          { value: 'SIMPLE', label: 'Simple', description: 'Just the essentials' },
          { value: 'STANDARD', label: 'Standard', description: 'A balanced amount of detail' },
          { value: 'DETAILED', label: 'Detailed', description: 'Every figure DiaLog can show' },
        ]}
      />

      <input type="hidden" name="displayName" value={profile.displayName ?? ''} />
      <input type="hidden" name="glucoseUnit" value={profile.glucoseUnit} />
      <input type="hidden" name="locale" value={profile.locale} />
      <input type="hidden" name="timezone" value={profile.timezone} />
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
      <HiddenBool name="aiEnabled" value={profile.aiEnabled} />
      <HiddenBool name="externalAiConsent" value={profile.externalAiConsentAt != null} />

      <div className="mt-5">
        <Submit />
      </div>
    </form>
  );
}
