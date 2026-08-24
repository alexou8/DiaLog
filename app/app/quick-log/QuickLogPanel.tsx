'use client';

import { useActionState } from 'react';
import { useFormStatus } from 'react-dom';
import { proposeQuickLogAction, type QuickLogState } from '@/lib/actions/assistant';
import { addExerciseAction, addMealAction, type RecordActionState } from '@/lib/actions/records';
import { Badge, Button, Card, CardHeader } from '@/components/ui';
import { Field, FormStatus, Select, TextArea, TextInput } from '@/components/ui/form';

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

/** Replace the time part of a `YYYY-MM-DDTHH:mm` value, keeping the date. */
function withTime(base: string, timeLocal: string | null): string {
  if (!timeLocal) return base;
  return `${base.slice(0, 10)}T${timeLocal}`;
}

function ProposeSubmit() {
  const { pending } = useFormStatus();
  return (
    <Button type="submit" disabled={pending} className="w-full">
      {pending ? 'Reading what you wrote…' : 'Suggest entries'}
    </Button>
  );
}

function SaveSubmit({ label }: { label: string }) {
  const { pending } = useFormStatus();
  return (
    <Button type="submit" disabled={pending}>
      {pending ? 'Saving…' : label}
    </Button>
  );
}

export function QuickLogPanel({ defaultTime }: { defaultTime: string }) {
  const [state, propose] = useActionState<QuickLogState | null, FormData>(proposeQuickLogAction, null);
  const [mealState, saveMeal] = useActionState<RecordActionState | null, FormData>(addMealAction, null);
  const [exerciseState, saveExercise] = useActionState<RecordActionState | null, FormData>(
    addExerciseAction,
    null,
  );

  return (
    <div className="space-y-6">
      <Card>
        <form action={propose} noValidate>
          <FormStatus status={state && !state.ok && state.message ? { ok: false, message: state.message } : null} />
          <Field
            label="What did you eat or do?"
            required
            hint="For example: “had a burger and fries around 7 and went for a 20 minute walk”."
            error={state?.errors?.text}
          >
            {({ id, describedBy, invalid }) => (
              <TextArea id={id} name="text" rows={3} required maxLength={600} aria-describedby={describedBy} invalid={invalid} />
            )}
          </Field>
          <ProposeSubmit />
        </form>
      </Card>

      <div aria-live="polite" className="space-y-6">
        {state?.ok && state.proposal ? (
          <>
            <h2 className="text-xl font-semibold">Check these before saving</h2>

            {state.proposal.meals.map((meal, index) => (
              <Card key={`meal-${index}`}>
                <CardHeader
                  level={3}
                  title="Suggested meal"
                  description="Change anything that is not right. The nutrition figures are estimates."
                  action={<Badge tone={CONFIDENCE_TONE[meal.confidence]}>{meal.confidence} confidence</Badge>}
                />
                <form action={saveMeal} noValidate>
                  <FormStatus
                    status={mealState && !mealState.ok && mealState.message ? { ok: false, message: mealState.message } : null}
                  />
                  <input type="hidden" name="estimateSource" value="AI_ESTIMATE" />

                  <Field label="What you ate" required>
                    {({ id }) => <TextInput id={id} name="description" defaultValue={meal.description} required />}
                  </Field>

                  <div className="grid gap-x-4 sm:grid-cols-2">
                    <Field label="When" required>
                      {({ id }) => (
                        <TextInput
                          id={id}
                          name="takenAt"
                          type="datetime-local"
                          required
                          defaultValue={withTime(defaultTime, meal.timeLocal)}
                        />
                      )}
                    </Field>
                    <Field label="Meal">
                      {({ id }) => (
                        <Select id={id} name="mealType" defaultValue={MEAL_TYPE_MAP[meal.mealType] ?? 'OTHER'}>
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
                      Worked out from your description. Treat these as rough figures and adjust them if
                      you know better.
                    </p>
                    <div className="grid grid-cols-2 gap-x-4 sm:grid-cols-3">
                      <Field label="Carbs (g)">
                        {({ id }) => (
                          <TextInput id={id} name="carbsG" type="number" step="1" inputMode="decimal" defaultValue={Math.round(meal.estimatedCarbsG)} />
                        )}
                      </Field>
                      <Field label="Protein (g)">
                        {({ id }) => (
                          <TextInput id={id} name="proteinG" type="number" step="1" inputMode="decimal" defaultValue={Math.round(meal.estimatedProteinG)} />
                        )}
                      </Field>
                      <Field label="Fat (g)">
                        {({ id }) => (
                          <TextInput id={id} name="fatG" type="number" step="1" inputMode="decimal" defaultValue={Math.round(meal.estimatedFatG)} />
                        )}
                      </Field>
                      <Field label="Fibre (g)">
                        {({ id }) => (
                          <TextInput id={id} name="fiberG" type="number" step="1" inputMode="decimal" defaultValue={Math.round(meal.estimatedFiberG)} />
                        )}
                      </Field>
                      <Field label="Calories">
                        {({ id }) => (
                          <TextInput id={id} name="calories" type="number" step="10" inputMode="decimal" defaultValue={Math.round(meal.estimatedCalories)} />
                        )}
                      </Field>
                    </div>
                  </fieldset>

                  <SaveSubmit label="Save this meal" />
                </form>
              </Card>
            ))}

            {state.proposal.exercise.map((session, index) => (
              <Card key={`exercise-${index}`}>
                <CardHeader
                  level={3}
                  title="Suggested activity"
                  action={<Badge tone={CONFIDENCE_TONE[session.confidence]}>{session.confidence} confidence</Badge>}
                />
                <form action={saveExercise} noValidate>
                  <FormStatus
                    status={
                      exerciseState && !exerciseState.ok && exerciseState.message
                        ? { ok: false, message: exerciseState.message }
                        : null
                    }
                  />
                  <Field label="Activity" required>
                    {({ id }) => <TextInput id={id} name="activity" defaultValue={session.activity} required />}
                  </Field>
                  <div className="grid gap-x-4 sm:grid-cols-3">
                    <Field label="When" required>
                      {({ id }) => (
                        <TextInput
                          id={id}
                          name="takenAt"
                          type="datetime-local"
                          required
                          defaultValue={withTime(defaultTime, session.timeLocal)}
                        />
                      )}
                    </Field>
                    <Field label="Minutes" required>
                      {({ id }) => (
                        <TextInput
                          id={id}
                          name="durationMin"
                          type="number"
                          inputMode="numeric"
                          required
                          defaultValue={Math.max(1, Math.round(session.durationMin))}
                        />
                      )}
                    </Field>
                    <Field label="How hard">
                      {({ id }) => (
                        <Select id={id} name="intensity" defaultValue={INTENSITY_MAP[session.intensity] ?? 'MODERATE'}>
                          <option value="LIGHT">Light</option>
                          <option value="MODERATE">Moderate</option>
                          <option value="VIGOROUS">Vigorous</option>
                        </Select>
                      )}
                    </Field>
                  </div>
                  <SaveSubmit label="Save this activity" />
                </form>
              </Card>
            ))}

            {state.proposal.meals.length === 0 && state.proposal.exercise.length === 0 ? (
              <Card>
                <p>
                  DiaLog could not turn that into entries it was confident about. You can rephrase it,
                  or use the ordinary forms, which are only slightly longer.
                </p>
              </Card>
            ) : null}

            {state.proposal.unparsed.length > 0 ? (
              <Card>
                <h3 className="font-semibold">Not turned into entries</h3>
                <p className="mt-1 text-ink-muted">
                  These parts were left alone rather than guessed at:
                </p>
                <ul className="mt-2 list-disc pl-5 text-ink-muted">
                  {state.proposal.unparsed.map((part, index) => (
                    <li key={index}>{part}</li>
                  ))}
                </ul>
              </Card>
            ) : null}
          </>
        ) : null}
      </div>
    </div>
  );
}
