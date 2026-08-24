'use client';

import { useActionState } from 'react';
import { useFormStatus } from 'react-dom';
import type { Profile } from '@prisma/client';
import { updatePreferencesAction, type ActionState } from '@/lib/actions/preferences';
import { fromMgdl } from '@/lib/domain/units';
import { Button, Callout } from '@/components/ui';
import { Checkbox, FormStatus } from '@/components/ui/form';
import { HiddenBool } from './HiddenBool';

function Submit() {
  const { pending } = useFormStatus();
  return (
    <Button type="submit" disabled={pending}>
      {pending ? 'Saving…' : 'Save assistant settings'}
    </Button>
  );
}

export function AssistantForm({
  profile,
  providerName,
  providerIsExternal,
}: {
  profile: Profile;
  providerName: string;
  providerIsExternal: boolean;
}) {
  const [state, action] = useActionState<ActionState | null, FormData>(updatePreferencesAction, null);

  return (
    <form action={action} noValidate className="rounded-[var(--radius-card)] border border-line bg-surface p-5 sm:p-6">
      <FormStatus status={state && state.message ? { ok: state.ok, message: state.message } : null} />

      <Checkbox
        name="aiEnabled"
        label="Turn the assistant on"
        description="When this is off, the chat assistant and its suggestions are hidden everywhere in DiaLog."
        defaultChecked={profile.aiEnabled}
      />

      <Callout tone={providerIsExternal ? 'notice' : 'positive'} icon={providerIsExternal ? '☁️' : '🔒'}>
        {providerIsExternal ? (
          <>
            <strong>Currently configured: {providerName} (external).</strong> When the assistant answers
            a question, a structured summary of your findings can be sent to this outside service to
            help generate the reply.
          </>
        ) : (
          <>
            <strong>Currently configured: {providerName} — nothing leaves this server.</strong> The
            assistant runs entirely on DiaLog&apos;s own infrastructure, so no external company ever sees
            your data.
          </>
        )}
      </Callout>

      {providerIsExternal ? (
        <div className="mt-4">
          <Checkbox
            name="externalAiConsent"
            label={`I agree to send structured findings to ${providerName}`}
            description="This can include summaries such as your average reading or how often you are in range. It never includes raw readings, meal descriptions, notes, or anything else word-for-word — and only when the assistant is on."
            defaultChecked={profile.externalAiConsentAt != null}
          />
          <p className="text-sm text-ink-muted">
            Withdrawing this consent switches the assistant to answer using only what it can compute
            from your data locally, without contacting {providerName}.
          </p>
        </div>
      ) : (
        <p className="mt-4 text-sm text-ink-muted">
          No external AI provider is configured on this deployment, so there is nothing external to
          consent to right now.
        </p>
      )}

      <input type="hidden" name="displayName" value={profile.displayName ?? ''} />
      <input type="hidden" name="glucoseUnit" value={profile.glucoseUnit} />
      <input type="hidden" name="locale" value={profile.locale} />
      <input type="hidden" name="timezone" value={profile.timezone} />
      <input type="hidden" name="targetLow" value={fromMgdl(profile.targetLowMgdl, profile.glucoseUnit)} />
      <input type="hidden" name="targetHigh" value={fromMgdl(profile.targetHighMgdl, profile.glucoseUnit)} />
      <input type="hidden" name="detailLevel" value={profile.detailLevel} />
      <HiddenBool name="largeText" value={profile.largeText} />
      <HiddenBool name="reduceMotion" value={profile.reduceMotion} />
      {!providerIsExternal ? <HiddenBool name="externalAiConsent" value={profile.externalAiConsentAt != null} /> : null}

      <div className="mt-5">
        <Submit />
      </div>
    </form>
  );
}
