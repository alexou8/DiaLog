'use client';

import { useActionState } from 'react';
import type { Profile } from '@prisma/client';
import { updatePreferencesAction, type ActionState } from '@/lib/actions/preferences';
import { fromMgdl } from '@/lib/domain/units';
import { Callout } from '@/components/ui';
import { ActionForm, Checkbox, SubmitButton } from '@/components/ui/form';
import { HiddenBool } from './HiddenBool';

export function AssistantForm({
  profile,
  providerName,
  providerIsExternal,
}: {
  profile: Profile;
  providerName: string;
  providerIsExternal: boolean;
}) {
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
      labels={{}}
    >
      <Checkbox
        name="aiEnabled"
        label="Turn the assistant on"
        description="When this is off, the chat assistant and its suggestions are hidden everywhere in DiaLog."
        defaultChecked={profile.aiEnabled}
      />

      <Callout
        tone={providerIsExternal ? 'notice' : 'positive'}
        icon={providerIsExternal ? 'external' : 'local'}
      >
        {providerIsExternal ? (
          <>
            <strong>Currently configured: {providerName} (external).</strong> Nothing is sent to it
            unless you agree below. Until then the assistant answers on this server with its
            built-in engine.
          </>
        ) : (
          <>
            <strong>Currently configured: {providerName}. Nothing leaves this server.</strong> The
            assistant runs entirely on DiaLog&apos;s own infrastructure, so no external company ever
            sees your data.
          </>
        )}
      </Callout>

      {providerIsExternal ? (
        <div className="mt-4">
          <Checkbox
            name="externalAiConsent"
            label={`I agree to send my questions and summaries to ${providerName}`}
            // Says exactly what leaves the server. The previous wording promised
            // meal descriptions were never sent word for word, which stopped
            // being true for consenting users: the pipeline sends the question
            // and quick-log text to the chosen provider verbatim.
            description="With this on, the questions you type and the text you enter in quick logging are sent word for word, together with summaries of your findings such as your average reading or time in range. Individual readings are never sent. With it off, nothing leaves this server and the built-in engine answers."
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
      {!providerIsExternal ? (
        <HiddenBool name="externalAiConsent" value={profile.externalAiConsentAt != null} />
      ) : null}

      <div className="mt-5">
        <SubmitButton>Save assistant settings</SubmitButton>
      </div>
    </ActionForm>
  );
}
