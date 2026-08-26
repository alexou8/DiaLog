'use client';

import { useActionState } from 'react';
import { useFormStatus } from 'react-dom';
import Link from 'next/link';
import { addMealAction, type RecordActionState } from '@/lib/actions/records';
import { toLocalInputValue } from '@/lib/domain/time';
import { Button, Icon } from '@/components/ui';
import { Field, FormStatus, Select, TextArea, TextInput } from '@/components/ui/form';

const MEAL_TYPES: { value: string; label: string }[] = [
  { value: 'BREAKFAST', label: 'Breakfast' },
  { value: 'LUNCH', label: 'Lunch' },
  { value: 'DINNER', label: 'Dinner' },
  { value: 'SNACK', label: 'Snack' },
  { value: 'OTHER', label: 'Other' },
];

function Submit() {
  const { pending } = useFormStatus();
  return (
    <Button type="submit" disabled={pending} className="w-full sm:w-auto">
      {pending ? 'Saving…' : 'Save meal'}
    </Button>
  );
}

export function MealForm({ timezone }: { timezone: string }) {
  const [state, action] = useActionState<RecordActionState | null, FormData>(addMealAction, null);
  const defaultTakenAt = toLocalInputValue(new Date(), timezone);

  return (
    <form action={action} noValidate>
      <FormStatus
        status={state && !state.ok && state.message ? { ok: false, message: state.message } : null}
      />

      <Callout />

      <Field
        label="What did you eat?"
        required
        error={state?.errors?.description}
        htmlFor="description"
      >
        {({ id, describedBy, invalid }) => (
          <TextArea
            id={id}
            name="description"
            required
            maxLength={300}
            placeholder="e.g. Two eggs, wholemeal toast, and an orange"
            aria-describedby={describedBy}
            invalid={invalid}
          />
        )}
      </Field>

      <Field label="When" required error={state?.errors?.takenAt} htmlFor="takenAt">
        {({ id, describedBy, invalid }) => (
          <TextInput
            id={id}
            name="takenAt"
            type="datetime-local"
            required
            defaultValue={defaultTakenAt}
            aria-describedby={describedBy}
            invalid={invalid}
          />
        )}
      </Field>

      <Field label="Meal type" error={state?.errors?.mealType} htmlFor="mealType">
        {({ id, describedBy, invalid }) => (
          <Select
            id={id}
            name="mealType"
            defaultValue="OTHER"
            aria-describedby={describedBy}
            invalid={invalid}
          >
            {MEAL_TYPES.map((m) => (
              <option key={m.value} value={m.value}>
                {m.label}
              </option>
            ))}
          </Select>
        )}
      </Field>

      <details className="mb-5 rounded-xl border border-line bg-surface-sunken">
        <summary className="cursor-pointer list-none px-4 py-3 text-base font-semibold text-brand-ink marker:content-none">
          <Icon name="add" />
          Add nutrition details (optional)
        </summary>
        <div className="border-t border-line p-4">
          <p className="mb-4 text-sm text-ink-muted">
            Leave any of these blank if you do not know them. Numbers are grams unless noted.
          </p>
          <div className="grid gap-4 sm:grid-cols-2">
            <Field label="Carbs" error={state?.errors?.carbsG} htmlFor="carbsG" hint="grams">
              {({ id, describedBy, invalid }) => (
                <TextInput
                  id={id}
                  name="carbsG"
                  type="number"
                  min={0}
                  step="any"
                  inputMode="decimal"
                  aria-describedby={describedBy}
                  invalid={invalid}
                />
              )}
            </Field>
            <Field label="Protein" error={state?.errors?.proteinG} htmlFor="proteinG" hint="grams">
              {({ id, describedBy, invalid }) => (
                <TextInput
                  id={id}
                  name="proteinG"
                  type="number"
                  min={0}
                  step="any"
                  inputMode="decimal"
                  aria-describedby={describedBy}
                  invalid={invalid}
                />
              )}
            </Field>
            <Field label="Fat" error={state?.errors?.fatG} htmlFor="fatG" hint="grams">
              {({ id, describedBy, invalid }) => (
                <TextInput
                  id={id}
                  name="fatG"
                  type="number"
                  min={0}
                  step="any"
                  inputMode="decimal"
                  aria-describedby={describedBy}
                  invalid={invalid}
                />
              )}
            </Field>
            <Field label="Fibre" error={state?.errors?.fiberG} htmlFor="fiberG" hint="grams">
              {({ id, describedBy, invalid }) => (
                <TextInput
                  id={id}
                  name="fiberG"
                  type="number"
                  min={0}
                  step="any"
                  inputMode="decimal"
                  aria-describedby={describedBy}
                  invalid={invalid}
                />
              )}
            </Field>
            <Field label="Calories" error={state?.errors?.calories} htmlFor="calories">
              {({ id, describedBy, invalid }) => (
                <TextInput
                  id={id}
                  name="calories"
                  type="number"
                  min={0}
                  step="any"
                  inputMode="decimal"
                  aria-describedby={describedBy}
                  invalid={invalid}
                />
              )}
            </Field>
            <Field
              label="Portion"
              error={state?.errors?.portion}
              htmlFor="portion"
              hint="e.g. 1 bowl, 200 ml"
            >
              {({ id, describedBy, invalid }) => (
                <TextInput
                  id={id}
                  name="portion"
                  maxLength={80}
                  aria-describedby={describedBy}
                  invalid={invalid}
                />
              )}
            </Field>
          </div>
          <Field label="Note" error={state?.errors?.note} htmlFor="note">
            {({ id, describedBy, invalid }) => (
              <TextArea
                id={id}
                name="note"
                maxLength={500}
                aria-describedby={describedBy}
                invalid={invalid}
              />
            )}
          </Field>
        </div>
      </details>

      <input type="hidden" name="estimateSource" value="USER_ENTERED" />

      <Submit />
    </form>
  );
}

function Callout() {
  return (
    <p className="mb-5 rounded-xl border border-line bg-surface-sunken p-4 text-sm text-ink-muted">
      In a hurry? You can{' '}
      <Link
        href="/app/quick-log"
        className="font-semibold text-brand-ink underline underline-offset-4"
      >
        type what you ate in your own words
      </Link>{' '}
      instead, and check what DiaLog suggests before saving.
    </p>
  );
}
