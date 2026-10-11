'use client';

import { startTransition, useActionState, useEffect, useRef, type FormEvent } from 'react';
import type { DetailLevel } from '@prisma/client';
import { askAssistantAction, type AssistantState } from '@/lib/actions/assistant';
import { Badge, Card, WhyThis } from '@/components/ui';
import { ActionForm, Field, Select, SubmitButton, TextArea } from '@/components/ui/form';
import { EmergencyAnswer } from '@/components/EmergencyAnswer';

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

export function AssistantPanel({ defaultDetail }: { defaultDetail: DetailLevel }) {
  const [state, action, pending] = useActionState<AssistantState | null, FormData>(
    askAssistantAction,
    null,
  );
  const formRef = useRef<HTMLFormElement>(null);
  const answerRef = useRef<HTMLHeadingElement>(null);
  const fallbackDetail = defaultDetail.toLowerCase();

  // A new answer takes focus, rather than the whole answer card sitting in a
  // live region: that re-read every paragraph, badge and disclosure on each
  // change. The short status line below is the only live region.
  // Not for an emergency answer: that one is a role="alert", which is
  // announced on its own, and focusing its heading as well read it twice.
  useEffect(() => {
    if (state?.ok && !state.emergency) answerRef.current?.focus();
  }, [state]);

  /**
   * Suggestion chips. Each is still its own <form> so it works without
   * JavaScript, but with it the chip is dispatched through the same action
   * state as the main form. Before, every chip form was invisible to the main
   * form's pending state: nothing changed during a multi-second answer, the
   * chips stayed clickable and people fired the same question repeatedly. The
   * chips also always posted the profile's default detail level, ignoring the
   * "How much detail" choice right above them; they now read it.
   */
  function askSuggestion(event: FormEvent<HTMLFormElement>, question: string) {
    event.preventDefault();
    if (pending) return;
    const chosen = formRef.current ? new FormData(formRef.current).get('detailLevel') : null;
    const formData = new FormData();
    formData.set('question', question);
    formData.set('detailLevel', typeof chosen === 'string' ? chosen : fallbackDetail);
    startTransition(() => action(formData));
  }

  const answer = state?.ok ? state.answer : undefined;
  const emergency = Boolean(state?.ok && state.emergency);

  return (
    <div className="flex flex-col gap-6">
      <Card>
        <ActionForm
          action={action}
          state={state}
          pending={pending}
          formRef={formRef}
          labels={{ question: 'Your question', detailLevel: 'How much detail would you like?' }}
        >
          <Field label="Your question" required name="question">
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
            name="detailLevel"
          >
            {({ id, describedBy }) => (
              <Select
                id={id}
                name="detailLevel"
                defaultValue={fallbackDetail}
                aria-describedby={describedBy}
              >
                <option value="simple">Simple: a short, plain answer</option>
                <option value="standard">Standard: the answer and the reasoning</option>
                <option value="detailed">Detailed: include the numbers</option>
              </Select>
            )}
          </Field>

          <SubmitButton className="w-full" pendingLabel="Looking at your data…">
            Ask
          </SubmitButton>
        </ActionForm>

        <p role="status" className="mt-3 text-sm text-ink-muted">
          {pending ? 'Looking at your data…' : ''}
        </p>

        {/* Not offered under an emergency answer: a row of casual questions
            next to "this may need help now" reads as if the safety message
            were optional. */}
        {emergency ? null : (
          <div className="mt-2">
            <h2 className="text-sm font-semibold text-ink-muted">Things people ask</h2>
            <ul className="mt-2 flex flex-wrap gap-2">
              {SUGGESTIONS.map((suggestion) => (
                <li key={suggestion}>
                  <form action={action} onSubmit={(event) => askSuggestion(event, suggestion)}>
                    <input type="hidden" name="question" value={suggestion} />
                    <input type="hidden" name="detailLevel" value={fallbackDetail} />
                    <button
                      type="submit"
                      disabled={pending}
                      className="dl-target rounded-[var(--radius-control)] border border-line-strong px-3 py-2 text-left text-sm hover:border-brand hover:text-brand-ink disabled:cursor-not-allowed disabled:opacity-60"
                    >
                      {suggestion}
                    </button>
                  </form>
                </li>
              ))}
            </ul>
          </div>
        )}
      </Card>

      {answer && emergency ? (
        <EmergencyAnswer
          answer={answer}
          headingId="assistant-answer"
          title="This may need help now, not an answer from your data"
        />
      ) : null}

      {state?.ok && answer && !emergency ? (
        <Card>
          <h2
            id="assistant-answer"
            ref={answerRef}
            tabIndex={-1}
            className="mb-3 text-xl font-semibold"
          >
            Your answer
          </h2>
          <p className="text-sm font-semibold text-ink-muted">You asked</p>
          <p className="mb-4 text-lg">{state.question}</p>

          <div className="flex flex-wrap items-center gap-2">
            <Badge tone={CONFIDENCE_TONE[answer.confidence]}>
              {CONFIDENCE_LABEL[answer.confidence]}
            </Badge>
            {answer.notEnoughData ? <Badge tone="neutral">Limited data</Badge> : null}
          </div>

          <p className="mt-3 text-lg font-semibold">{answer.shortAnswer}</p>

          {answer.detail.length > 0 ? (
            <div className="mt-3 flex flex-col gap-2 text-ink-muted">
              {answer.detail.map((paragraph, index) => (
                <p key={index}>{paragraph}</p>
              ))}
            </div>
          ) : null}

          {answer.suggestedQuestionsForClinician.length > 0 ? (
            <div className="mt-5 rounded-[var(--radius-card)] border border-line bg-surface-sunken p-4">
              <h3 className="font-semibold">Worth raising with your healthcare professional</h3>
              <ul className="mt-2 flex list-disc flex-col gap-1 pl-5 text-ink-muted">
                {answer.suggestedQuestionsForClinician.map((question, index) => (
                  <li key={index}>{question}</li>
                ))}
              </ul>
            </div>
          ) : null}

          <WhyThis label="What was this answer based on?">
            {state.citedFindings && state.citedFindings.length > 0 ? (
              <ul className="flex flex-col gap-3">
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
                No specific finding was strong enough to cite, which is why the answer is cautious.
                More logged days will change that.
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
  );
}
