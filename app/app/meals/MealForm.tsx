'use client';

import { useActionState } from 'react';
import Link from 'next/link';
import { addMealAction, type RecordActionState } from '@/lib/actions/records';
import { toLocalInputValue } from '@/lib/domain/time';
import { Icon } from '@/components/ui';
import { ActionForm, Field, Select, SubmitButton, TextArea, TextInput } from '@/components/ui/form';

const MEAL_TYPES: { value: string; label: string }[] = [
  { value: 'BREAKFAST', label: 'Breakfast' },
  { value: 'LUNCH', label: 'Lunch' },
  { value: 'DINNER', label: 'Dinner' },
  { value: 'SNACK', label: 'Snack' },
  { value: 'OTHER', label: 'Other' },
];

export function MealForm({ timezone }: { timezone: string }) {
  const [state, action, pending] = useActionState<RecordActionState | null, FormData>(
    addMealAction,
    null,
  );
  const defaultTakenAt = toLocalInputValue(new Date(), timezone);

  return (
    <ActionForm
      action={action}
      state={state}
      pending={pending}
      labels={{
        description: 'What did you eat?',
        takenAt: 'When',
        mealType: 'Meal type',
        carbsG: 'Carbs',
        proteinG: 'Protein',
        fatG: 'Fat',
        fiberG: 'Fibre',
        calories: 'Calories',
        portion: 'Portion',
        note: 'Note',
      }}
    >
      <Callout />

      <Field label="What did you eat?" required name="description">
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

      <Field label="When" required name="takenAt">
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

      <Field label="Meal type" name="mealType">
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

      <details className="mb-5 rounded-[var(--radius-card)] border border-line bg-surface-sunken">
        <summary className="cursor-pointer list-none px-4 py-3 text-base font-semibold text-brand-ink marker:content-none">
          <Icon name="add" />
          Add nutrition details (optional)
        </summary>
        <div className="border-t border-line p-4">
          <p className="mb-4 text-sm text-ink-muted">
            Leave any of these blank if you do not know them. Numbers are grams unless noted.
          </p>
          <div className="grid gap-4 sm:grid-cols-2">
            <Field label="Carbs" name="carbsG" hint="grams">
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
            <Field label="Protein" name="proteinG" hint="grams">
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
            <Field label="Fat" name="fatG" hint="grams">
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
            <Field label="Fibre" name="fiberG" hint="grams">
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
            <Field label="Calories" name="calories">
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
            <Field label="Portion" name="portion" hint="e.g. 1 bowl, 200 ml">
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
          <Field label="Note" name="note">
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

      <SubmitButton className="w-full sm:w-auto">Save meal</SubmitButton>
    </ActionForm>
  );
}

function Callout() {
  return (
    <p className="mb-5 rounded-[var(--radius-card)] border border-line bg-surface-sunken p-4 text-sm text-ink-muted">
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
