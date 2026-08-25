'use client';

import { useActionState } from 'react';
import { useFormStatus } from 'react-dom';
import type { DetailLevel } from '@prisma/client';
import { askAssistantAction, type AssistantState } from '@/lib/actions/assistant';
import { Badge, Button, Card, WhyThis } from '@/components/ui';
import { Field, FormStatus, Select, TextArea } from '@/components/ui/form';

const SUGGESTIONS = [
  'What patterns have you noticed recently?',
  'Why were my readings higher this week?',
  'Does walking after dinner seem to help me?',
  'Which meals seem to affect my glucose most?',
  'Summarise my past month.',
  'What should I ask my doctor about?',
];

const CONFIDENCE_LABEL = {
  insufficient: 'Not enough evidence',
  low: 'Low confidence',
  moderate: 'Moderate confidence',
  high: 'High confidence',
} as const;

const CONFIDENCE_TONE = {
  insufficient: 'neutral',
  low: 'info',
  moderate: 'brand',
  high: 'positive',
} as const;

function Submit() {
  const { pending } = useFormStatus();
  return (
    <Button type="submit" disabled={pending} className="w-full">
      {pending ? 'Looking at your data…' : 'Ask'}
    </Button>
  );
}

export function AssistantPanel({ defaultDetail }: { defaultDetail: DetailLevel }) {
  const [state, action] = useActionState<AssistantState | null, FormData>(askAssistantAction, null);

  return (
    <div className="space-y-6">
      <Card>
        <form action={action} noValidate>
          <FormStatus
            status={
              state && !state.ok && state.message ? { ok: false, message: state.message } : null
            }
          />

          <Field label="Your question" required error={state?.errors?.question}>
            {({ id, describedBy, invalid }) => (
              <TextArea
                id={id}
                name="question"
                rows={3}
                required
                maxLength={500}
                placeholder="For example: have my mornings been steadier this month?"
                aria-describedby={describedBy}
                invalid={invalid}
              />
            )}
          </Field>

          <Field
            label="How much detail would you like?"
            hint="You can change this for any question."
          >
            {({ id }) => (
              <Select id={id} name="detailLevel" defaultValue={defaultDetail.toLowerCase()}>
                <option value="simple">Simple — a short, plain answer</option>
                <option value="standard">Standard — the answer and the reasoning</option>
                <option value="detailed">Detailed — include the numbers</option>
              </Select>
            )}
          </Field>

          <Submit />
        </form>

        <div className="mt-5">
          <h2 className="text-sm font-semibold text-ink-muted">Things people ask</h2>
          <ul className="mt-2 flex flex-wrap gap-2">
            {SUGGESTIONS.map((suggestion) => (
              <li key={suggestion}>
                <form action={action}>
                  <input type="hidden" name="question" value={suggestion} />
                  <input type="hidden" name="detailLevel" value={defaultDetail.toLowerCase()} />
                  <button
                    type="submit"
                    className="dl-target rounded-full border border-line-strong px-3 py-2 text-sm hover:border-brand hover:text-brand-ink"
                  >
                    {suggestion}
                  </button>
                </form>
              </li>
            ))}
          </ul>
        </div>
      </Card>

      <div aria-live="polite">
        {state?.ok && state.answer ? (
          <Card>
            <p className="text-sm font-semibold text-ink-muted">You asked</p>
            <p className="mb-4 text-lg">{state.question}</p>

            <div className="flex flex-wrap items-center gap-2">
              <Badge tone={CONFIDENCE_TONE[state.answer.confidence]}>
                {CONFIDENCE_LABEL[state.answer.confidence]}
              </Badge>
              {state.answer.notEnoughData ? <Badge tone="neutral">Limited data</Badge> : null}
            </div>

            <p className="mt-3 text-lg font-semibold">{state.answer.shortAnswer}</p>

            {state.answer.detail.length > 0 ? (
              <div className="mt-3 space-y-2 text-ink-muted">
                {state.answer.detail.map((paragraph, index) => (
                  <p key={index}>{paragraph}</p>
                ))}
              </div>
            ) : null}

            {state.answer.suggestedQuestionsForClinician.length > 0 ? (
              <div className="mt-5 rounded-xl border border-line bg-surface-sunken p-4">
                <h3 className="font-semibold">Worth raising with your healthcare professional</h3>
                <ul className="mt-2 list-disc space-y-1 pl-5 text-ink-muted">
                  {state.answer.suggestedQuestionsForClinician.map((question, index) => (
                    <li key={index}>{question}</li>
                  ))}
                </ul>
              </div>
            ) : null}

            <WhyThis label="What was this answer based on?">
              {state.citedFindings && state.citedFindings.length > 0 ? (
                <ul className="space-y-3">
                  {state.citedFindings.map((finding) => (
                    <li key={finding.id}>
                      <p className="font-medium text-ink">{finding.statement}</p>
                      <p>
                        {finding.basis} Based on {finding.sampleSize}{' '}
                        {finding.sampleSize === 1 ? 'record' : 'records'} (
                        {finding.evidenceLevel.toLowerCase()}).
                      </p>
                    </li>
                  ))}
                </ul>
              ) : (
                <p>
                  No specific finding was strong enough to cite, which is why the answer is
                  cautious. More logged days will change that.
                </p>
              )}
              <p className="mt-3 text-xs">
                Answer produced by: {state.providerId}
                {state.usedFallback ? ' (fell back to the built-in engine)' : ''}.
              </p>
            </WhyThis>
          </Card>
        ) : null}
      </div>
    </div>
  );
}
