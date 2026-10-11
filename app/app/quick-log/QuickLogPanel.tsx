'use client';

import {
  useActionState,
  useCallback,
  useEffect,
  useRef,
  useState,
  useSyncExternalStore,
} from 'react';
import Link from 'next/link';
import { proposeQuickLogAction, type QuickLogState } from '@/lib/actions/assistant';
import { addExerciseAction, addMealAction, type RecordActionState } from '@/lib/actions/records';
import type { MealParse } from '@/lib/ai/schemas';
import { Badge, Button, Card, CardHeader, Icon } from '@/components/ui';
import { ActionForm, Field, Select, SubmitButton, TextArea, TextInput } from '@/components/ui/form';
import { EmergencyAnswer } from '@/components/EmergencyAnswer';

const MEAL_TYPE_MAP: Record<string, string> = {
  breakfast: 'BREAKFAST',
  lunch: 'LUNCH',
  dinner: 'DINNER',
  snack: 'SNACK',
  unknown: 'OTHER',
};

const INTENSITY_MAP: Record<string, string> = {
  light: 'LIGHT',
  moderate: 'MODERATE',
  vigorous: 'VIGOROUS',
};

const CONFIDENCE_TONE = { low: 'caution', medium: 'notice', high: 'positive' } as const;

type CardStatus = 'unsaved' | 'saving' | 'saved' | 'problem';

const STATUS_BADGE: Record<
  CardStatus,
  {
    tone: 'neutral' | 'info' | 'positive' | 'critical';
    icon: 'info' | 'ok' | 'caution';
    label: string;
  }
> = {
  unsaved: { tone: 'neutral', icon: 'info', label: 'Not saved yet' },
  saving: { tone: 'info', icon: 'info', label: 'Saving…' },
  saved: { tone: 'positive', icon: 'ok', label: 'Saved' },
  problem: { tone: 'critical', icon: 'caution', label: 'Needs a fix' },
};

/** Replace the time part of a `YYYY-MM-DDTHH:mm` value, keeping the date. */
function withTime(base: string, timeLocal: string | null): string {
  if (!timeLocal) return base;
  return `${base.slice(0, 10)}T${timeLocal}`;
}

const noopSubscribe = () => () => {};

/** False during SSR and hydration, true once React owns the page. */
function useHydrated(): boolean {
  return useSyncExternalStore(
    noopSubscribe,
    () => true,
    () => false,
  );
}

function statusOf(state: RecordActionState | null, pending: boolean): CardStatus {
  if (pending) return 'saving';
  if (!state) return 'unsaved';
  return state.ok ? 'saved' : 'problem';
}

interface CardProps {
  id: string;
  defaultTime: string;
  register: (id: string, form: HTMLFormElement | null) => void;
  onStatus: (id: string, status: CardStatus) => void;
  /** Whether a failed save may move focus to this card's error summary. */
  shouldFocusSummary: () => boolean;
}

/**
 * One suggestion, with its own action state. All meal cards used to share a
 * single `useActionState`, and the save action redirected: saving the first
 * suggestion of "burger at 7 and a 20-minute walk" navigated to /app/meals and
 * the walk was gone. Each card now saves in place (see `_result` below) and
 * reports its own status.
 */
function useProposalCard(
  action: typeof addMealAction,
  { id, register, onStatus }: Omit<CardProps, 'defaultTime'>,
) {
  const [state, dispatch, pending] = useActionState<RecordActionState | null, FormData>(
    action,
    null,
  );
  const status = statusOf(state, pending);
  const formRef = useRef<HTMLFormElement>(null);
  const hydrated = useHydrated();

  useEffect(() => {
    onStatus(id, status);
  }, [id, status, onStatus]);

  // Saved cards drop out of "Save all".
  useEffect(() => {
    register(id, status === 'saved' ? null : formRef.current);
    return () => register(id, null);
  }, [id, status, register]);

  return { state, dispatch, pending, status, formRef, hydrated };
}

function CardStatusBadge({ status }: { status: CardStatus }) {
  const badge = STATUS_BADGE[status];
  return (
    <Badge tone={badge.tone} icon={badge.icon}>
      {badge.label}
    </Badge>
  );
}

/**
 * Asks the record action for an in-place result instead of a redirect. Only
 * rendered once hydrated: without JavaScript there is no client state to keep
 * the other suggestions in, so the ordinary redirect is the better outcome.
 */
function InlineResultField({ hydrated }: { hydrated: boolean }) {
  return hydrated ? <input type="hidden" name="_result" value="inline" /> : null;
}

function MealProposalCard({
  meal,
  defaultTime,
  ...card
}: CardProps & { meal: MealParse['meals'][number] }) {
  const { state, dispatch, pending, status, formRef, hydrated } = useProposalCard(
    addMealAction,
    card,
  );

  return (
    <Card as="li">
      <CardHeader
        level={3}
        title={meal.description || 'Suggested meal'}
        description="Change anything that is not right. The nutrition figures are estimates."
        action={
          <div className="flex flex-wrap gap-2">
            <CardStatusBadge status={status} />
            <Badge tone={CONFIDENCE_TONE[meal.confidence]}>{meal.confidence} confidence</Badge>
          </div>
        }
      />
      {status === 'saved' ? (
        <p className="mb-5 flex items-center gap-2">
          <Icon name="ok" className="shrink-0 text-positive" />
          <span>
            Saved to{' '}
            <Link href="/app/meals" className="font-semibold underline underline-offset-4">
              your meals
            </Link>
            .
          </span>
        </p>
      ) : (
        <ActionForm
          action={dispatch}
          state={state}
          pending={pending}
          formRef={formRef}
          showSuccess={false}
          shouldFocusSummary={card.shouldFocusSummary}
          labels={{
            description: 'What you ate',
            takenAt: 'When',
            mealType: 'Meal',
            carbsG: 'Carbs (g)',
            proteinG: 'Protein (g)',
            fatG: 'Fat (g)',
            fiberG: 'Fibre (g)',
            calories: 'Calories',
          }}
        >
          <input type="hidden" name="estimateSource" value="AI_ESTIMATE" />
          <InlineResultField hydrated={hydrated} />

          <Field label="What you ate" required name="description">
            {({ id, describedBy, invalid }) => (
              <TextInput
                id={id}
                name="description"
                defaultValue={meal.description}
                required
                aria-describedby={describedBy}
                invalid={invalid}
              />
            )}
          </Field>

          <div className="grid gap-x-4 sm:grid-cols-2">
            <Field label="When" required name="takenAt">
              {({ id, describedBy, invalid }) => (
                <TextInput
                  id={id}
                  name="takenAt"
                  type="datetime-local"
                  required
                  defaultValue={withTime(defaultTime, meal.timeLocal)}
                  aria-describedby={describedBy}
                  invalid={invalid}
                />
              )}
            </Field>
            <Field label="Meal" name="mealType">
              {({ id, describedBy, invalid }) => (
                <Select
                  id={id}
                  name="mealType"
                  defaultValue={MEAL_TYPE_MAP[meal.mealType] ?? 'OTHER'}
                  aria-describedby={describedBy}
                  invalid={invalid}
                >
                  <option value="BREAKFAST">Breakfast</option>
                  <option value="LUNCH">Lunch</option>
                  <option value="DINNER">Dinner</option>
                  <option value="SNACK">Snack</option>
                  <option value="OTHER">Other</option>
                </Select>
              )}
            </Field>
          </div>

          <fieldset className="mb-5">
            <legend className="mb-2 text-base font-semibold">Estimated nutrition</legend>
            <p className="mb-3 text-sm text-ink-muted">
              Worked out from your description. Treat these as rough figures and adjust them if you
              know better.
            </p>
            <div className="grid grid-cols-2 gap-x-4 sm:grid-cols-3">
              {(
                [
                  ['carbsG', 'Carbs (g)', meal.estimatedCarbsG, '1'],
                  ['proteinG', 'Protein (g)', meal.estimatedProteinG, '1'],
                  ['fatG', 'Fat (g)', meal.estimatedFatG, '1'],
                  ['fiberG', 'Fibre (g)', meal.estimatedFiberG, '1'],
                  ['calories', 'Calories', meal.estimatedCalories, '10'],
                ] as const
              ).map(([name, label, estimate, step]) => (
                <Field key={name} label={label} name={name}>
                  {({ id, describedBy, invalid }) => (
                    <TextInput
                      id={id}
                      name={name}
                      type="number"
                      min={0}
                      step={step}
                      inputMode="decimal"
                      defaultValue={Math.round(estimate)}
                      className="dl-numeric"
                      aria-describedby={describedBy}
                      invalid={invalid}
                    />
                  )}
                </Field>
              ))}
            </div>
          </fieldset>

          <SubmitButton>Save this meal</SubmitButton>
        </ActionForm>
      )}
    </Card>
  );
}

function ExerciseProposalCard({
  session,
  defaultTime,
  ...card
}: CardProps & { session: MealParse['exercise'][number] }) {
  const { state, dispatch, pending, status, formRef, hydrated } = useProposalCard(
    addExerciseAction,
    card,
  );

  return (
    <Card as="li">
      <CardHeader
        level={3}
        title={session.activity || 'Suggested activity'}
        action={
          <div className="flex flex-wrap gap-2">
            <CardStatusBadge status={status} />
            <Badge tone={CONFIDENCE_TONE[session.confidence]}>
              {session.confidence} confidence
            </Badge>
          </div>
        }
      />
      {status === 'saved' ? (
        <p className="mb-5 flex items-center gap-2">
          <Icon name="ok" className="shrink-0 text-positive" />
          <span>
            Saved to{' '}
            <Link href="/app/activity" className="font-semibold underline underline-offset-4">
              your activity
            </Link>
            .
          </span>
        </p>
      ) : (
        <ActionForm
          action={dispatch}
          state={state}
          pending={pending}
          formRef={formRef}
          showSuccess={false}
          shouldFocusSummary={card.shouldFocusSummary}
          labels={{
            activity: 'Activity',
            takenAt: 'When',
            durationMin: 'Minutes',
            intensity: 'How hard',
          }}
        >
          <InlineResultField hydrated={hydrated} />
          <Field label="Activity" required name="activity">
            {({ id, describedBy, invalid }) => (
              <TextInput
                id={id}
                name="activity"
                defaultValue={session.activity}
                required
                aria-describedby={describedBy}
                invalid={invalid}
              />
            )}
          </Field>
          <div className="grid gap-x-4 sm:grid-cols-3">
            <Field label="When" required name="takenAt">
              {({ id, describedBy, invalid }) => (
                <TextInput
                  id={id}
                  name="takenAt"
                  type="datetime-local"
                  required
                  defaultValue={withTime(defaultTime, session.timeLocal)}
                  aria-describedby={describedBy}
                  invalid={invalid}
                />
              )}
            </Field>
            <Field label="Minutes" required name="durationMin">
              {({ id, describedBy, invalid }) => (
                <TextInput
                  id={id}
                  name="durationMin"
                  type="number"
                  inputMode="numeric"
                  required
                  defaultValue={Math.max(1, Math.round(session.durationMin))}
                  className="dl-numeric"
                  aria-describedby={describedBy}
                  invalid={invalid}
                />
              )}
            </Field>
            <Field label="How hard" name="intensity">
              {({ id, describedBy, invalid }) => (
                <Select
                  id={id}
                  name="intensity"
                  defaultValue={INTENSITY_MAP[session.intensity] ?? 'MODERATE'}
                  aria-describedby={describedBy}
                  invalid={invalid}
                >
                  <option value="LIGHT">Light</option>
                  <option value="MODERATE">Moderate</option>
                  <option value="VIGOROUS">Vigorous</option>
                </Select>
              )}
            </Field>
          </div>
          <SubmitButton>Save this activity</SubmitButton>
        </ActionForm>
      )}
    </Card>
  );
}

/**
 * Quick-log as a review queue: the original text stays in its box above, and
 * each suggestion below is a card with its own status, saved one at a time or
 * all at once. Nothing is saved until the person presses a save button.
 */
export function QuickLogPanel({ defaultTime }: { defaultTime: string }) {
  const [state, propose, proposing] = useActionState<QuickLogState | null, FormData>(
    proposeQuickLogAction,
    null,
  );

  // The result whose suggestions are on screen. Only a new set of
  // suggestions (or a safety answer) replaces it, and that resets the queue:
  // new card keys, so no saved state carries over to a different suggestion,
  // and no remembered statuses. A failed resubmit (an emptied box, the rate
  // limit) used to reset it too and threw away every unsaved card; its error
  // now shows on the form above and the cards stay.
  const [seenState, setSeenState] = useState(state);
  const [shown, setShown] = useState<QuickLogState | null>(() => (state?.ok ? state : null));
  const [round, setRound] = useState(0);
  const [statuses, setStatuses] = useState<Record<string, CardStatus>>({});
  if (state !== seenState) {
    setSeenState(state);
    if (state?.ok && (state.proposal || state.emergency)) {
      setShown(state);
      setRound((r) => r + 1);
      setStatuses({});
    }
  }

  const forms = useRef(new Map<string, HTMLFormElement>());
  const register = useCallback((id: string, form: HTMLFormElement | null) => {
    if (form) forms.current.set(id, form);
    else forms.current.delete(id);
  }, []);
  const onStatus = useCallback((id: string, status: CardStatus) => {
    setStatuses((current) => (current[id] === status ? current : { ...current, [id]: status }));
  }, []);

  const headingRef = useRef<HTMLHeadingElement>(null);
  const listRef = useRef<HTMLUListElement>(null);
  // Text that described a possible emergency gets the fixed safety answer
  // instead of suggestion cards: that result is about getting help, not about
  // what to log.
  const emergencyAnswer = shown?.emergency ? shown.emergencyAnswer : undefined;
  const proposal = shown && !shown.emergency ? shown.proposal : undefined;
  const total = proposal ? proposal.meals.length + proposal.exercise.length : 0;

  // Focus moves to the results heading when suggestions arrive, instead of
  // the whole results area being a live region (which re-read every editable
  // field on each change). Not for a safety answer: it is a role="alert" and
  // is announced on its own.
  useEffect(() => {
    if (state?.ok && state.proposal && !state.emergency) headingRef.current?.focus();
  }, [state]);

  const values = Object.values(statuses);
  const saved = values.filter((s) => s === 'saved').length;
  const saving = values.some((s) => s === 'saving');
  const remaining = total - saved;

  // "Save all" submits every card at once. Each failing card used to move
  // focus to its own error summary, so focus jumped from card to card and
  // ended on whichever answered last. During a batch the cards leave focus
  // alone; once every save has settled it goes to the first card that needs a
  // fix, in reading order, or to the heading when everything saved.
  const batch = useRef<'idle' | 'requested' | 'running'>('idle');
  const shouldFocusSummary = useCallback(() => batch.current === 'idle', []);
  useEffect(() => {
    if (saving) {
      if (batch.current === 'requested') batch.current = 'running';
      return;
    }
    if (batch.current !== 'running') return;
    batch.current = 'idle';
    const firstProblem = listRef.current?.querySelector<HTMLElement>(
      '[role="alert"][tabindex="-1"]',
    );
    (firstProblem ?? headingRef.current)?.focus();
  }, [saving]);

  function saveAll() {
    // Each card's own form is submitted, so each goes through the same
    // validation, pending state and error summary as saving it by hand.
    batch.current = 'requested';
    for (const form of forms.current.values()) form.requestSubmit();
  }

  let statusLine = '';
  if (proposing) statusLine = 'Reading what you wrote…';
  else if (total > 0 && saved === total)
    statusLine = `All ${total} ${total === 1 ? 'entry is' : 'entries are'} saved.`;
  else if (total > 0 && saved > 0) statusLine = `${saved} of ${total} saved.`;
  else if (total > 0)
    statusLine = `${total} ${total === 1 ? 'suggestion' : 'suggestions'} ready to check.`;

  return (
    <div className="flex flex-col gap-6">
      <Card>
        <ActionForm
          action={propose}
          state={state}
          pending={proposing}
          labels={{ text: 'What did you eat or do?' }}
        >
          <Field
            label="What did you eat or do?"
            required
            hint="For example: “had a burger and fries around 7 and went for a 20 minute walk”."
            name="text"
          >
            {({ id, describedBy, invalid }) => (
              <TextArea
                id={id}
                name="text"
                rows={3}
                required
                maxLength={600}
                aria-describedby={describedBy}
                invalid={invalid}
              />
            )}
          </Field>
          <SubmitButton className="w-full" pendingLabel="Reading what you wrote…">
            Suggest entries
          </SubmitButton>
        </ActionForm>
      </Card>

      {/* The one live region: a short status line, never the cards. */}
      <p role="status" className="sr-only">
        {statusLine}
      </p>

      {emergencyAnswer ? (
        <EmergencyAnswer
          answer={emergencyAnswer}
          headingId="quick-log-emergency"
          title="This may need help now, before anything is logged"
        />
      ) : null}

      {proposal ? (
        <section aria-labelledby="quick-log-results" className="flex flex-col gap-4">
          <div className="flex flex-wrap items-end justify-between gap-3">
            <div>
              <h2
                id="quick-log-results"
                ref={headingRef}
                tabIndex={-1}
                className="text-xl font-semibold"
              >
                Check these before saving
              </h2>
              {total > 0 ? (
                <p className="mt-1 text-ink-muted" aria-hidden="true">
                  {statusLine}
                </p>
              ) : null}
            </div>
            {remaining > 1 ? (
              <Button type="button" onClick={saveAll} disabled={saving}>
                <Icon name="ok" />
                {saving ? 'Saving…' : `Save all ${remaining}`}
              </Button>
            ) : null}
          </div>

          {total > 0 ? (
            <ul key={round} ref={listRef} className="flex flex-col gap-4">
              {proposal.meals.map((meal, index) => (
                <MealProposalCard
                  key={`meal-${index}`}
                  id={`${round}-meal-${index}`}
                  meal={meal}
                  defaultTime={defaultTime}
                  register={register}
                  onStatus={onStatus}
                  shouldFocusSummary={shouldFocusSummary}
                />
              ))}
              {proposal.exercise.map((session, index) => (
                <ExerciseProposalCard
                  key={`exercise-${index}`}
                  id={`${round}-exercise-${index}`}
                  session={session}
                  defaultTime={defaultTime}
                  register={register}
                  onStatus={onStatus}
                  shouldFocusSummary={shouldFocusSummary}
                />
              ))}
            </ul>
          ) : (
            <Card>
              <p>
                DiaLog could not turn that into entries it was confident about. You can rephrase it,
                or use the ordinary forms, which are only slightly longer.
              </p>
            </Card>
          )}

          {proposal.unparsed.length > 0 ? (
            <Card>
              <h3 className="font-semibold">Not turned into entries</h3>
              <p className="mt-1 text-ink-muted">
                These parts were left alone rather than guessed at:
              </p>
              <ul className="mt-2 list-disc pl-5 text-ink-muted">
                {proposal.unparsed.map((part, index) => (
                  <li key={index}>{part}</li>
                ))}
              </ul>
            </Card>
          ) : null}
        </section>
      ) : null}
    </div>
  );
}
