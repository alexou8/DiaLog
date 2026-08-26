/**
 * Accessible form primitives, built on shadcn/ui.
 *
 * Health data entry is the part of this product where a mistake is most
 * expensive, so the rules here are stricter than the component library's
 * defaults and are enforced by the primitives rather than left to call sites:
 *
 * - Every field is a real <label> bound to a real control. Placeholder text is
 *   never a substitute for a label.
 * - Required *and* optional are stated in words. "No asterisk" is not an
 *   answer to "is this one optional?".
 * - Errors are associated with `aria-describedby`, announced through a live
 *   region, and written in plain language ("Please enter a number between 1.1
 *   and 38.9") rather than as validation jargon.
 * - Hints sit above the control, not in a tooltip. Nothing a person needs in
 *   order to answer correctly is hidden behind a hover.
 */
'use client';

import { useId, type ReactNode } from 'react';

import { cn } from '@/lib/utils';
import { Icon } from './icon';
import { Input } from './input';
import { Label } from './label';
import { Textarea } from './textarea';

export function Field({
  label,
  hint,
  error,
  children,
  required,
  htmlFor,
}: {
  label: string;
  hint?: ReactNode;
  error?: string;
  children: (ids: { id: string; describedBy: string | undefined; invalid: boolean }) => ReactNode;
  required?: boolean;
  htmlFor?: string;
}) {
  const generated = useId();
  const id = htmlFor ?? generated;
  const hintId = hint ? `${id}-hint` : undefined;
  const errorId = error ? `${id}-error` : undefined;
  const describedBy = [hintId, errorId].filter(Boolean).join(' ') || undefined;

  return (
    <div className="mb-5">
      <Label htmlFor={id} className="mb-1.5">
        {label}
        <span className="font-normal text-ink-muted">{required ? '(required)' : '(optional)'}</span>
      </Label>
      {hint ? (
        <p id={hintId} className="dl-measure mb-2 text-sm text-ink-muted">
          {hint}
        </p>
      ) : null}
      {children({ id, describedBy, invalid: Boolean(error) })}
      {error ? (
        <p
          id={errorId}
          className="mt-1.5 flex items-center gap-1.5 text-sm font-medium text-critical"
        >
          <Icon name="caution" className="shrink-0" />
          {error}
        </p>
      ) : null}
    </div>
  );
}

export function TextInput({
  invalid,
  ...rest
}: React.ComponentProps<typeof Input> & { invalid?: boolean }) {
  return <Input {...rest} aria-invalid={invalid || undefined} />;
}

export function TextArea({
  invalid,
  ...rest
}: React.ComponentProps<typeof Textarea> & { invalid?: boolean }) {
  return <Textarea {...rest} aria-invalid={invalid || undefined} />;
}

/**
 * A native <select>, styled with the shadcn control tokens.
 *
 * Deliberately not shadcn's Radix Select. These forms post to Server Actions
 * and must work before hydration; a Radix Select renders a button plus a
 * portalled listbox and submits nothing without JavaScript. The native control
 * also gets the platform's own picker, which on a phone is a larger and more
 * familiar target than anything rendered in the page.
 *
 * The same reasoning applies to RadioCards and Checkbox below. shadcn's Select,
 * RadioGroup and Checkbox are used elsewhere in the product, just not for
 * controls whose value has to survive a submit that beats hydration.
 */
export function Select({
  invalid,
  className,
  children,
  ...rest
}: React.SelectHTMLAttributes<HTMLSelectElement> & { invalid?: boolean }) {
  return (
    <select
      {...rest}
      aria-invalid={invalid || undefined}
      className={cn(
        'min-h-11 w-full rounded-[var(--radius-control)] border-2 border-input bg-surface px-4 py-2.5 text-base text-ink outline-none transition-colors focus:border-brand aria-invalid:border-destructive',
        className,
      )}
    >
      {children}
    </select>
  );
}

/**
 * Large tappable radio group.
 *
 * Native <input type="radio"> for the reason given on `Select` above: Radix's
 * RadioGroup carries its value in a hidden input that is only attached once
 * the component finds its <form> on the client, so a submit that beats
 * hydration posts nothing. These forms are progressively enhanced through
 * `useActionState`, and a settings change that silently saves the wrong value
 * is worse than a plainer control.
 *
 * Used instead of a dropdown wherever there are few options: a visible set of
 * big targets is easier than a picker for someone with limited dexterity or
 * eyesight, and it shows every option's description without opening anything.
 * `has-[:checked]` gives the selected card a border and a fill, so the choice
 * is not signalled by the small radio dot alone.
 */
export function RadioCards<T extends string>({
  name,
  legend,
  hint,
  options,
  defaultValue,
  columns = 2,
}: {
  name: string;
  legend: string;
  hint?: string;
  options: { value: T; label: string; description?: string }[];
  defaultValue?: T;
  columns?: 1 | 2 | 3;
}) {
  const id = useId();
  const hintId = hint ? `${id}-hint` : undefined;

  return (
    <fieldset className="mb-5" aria-describedby={hintId}>
      <legend className="mb-1.5 text-base font-semibold">{legend}</legend>
      {hint ? (
        <p id={hintId} className="dl-measure mb-2 text-sm text-ink-muted">
          {hint}
        </p>
      ) : null}
      <div
        className={cn(
          'grid gap-2',
          columns === 1 && 'grid-cols-1',
          columns === 2 && 'grid-cols-1 sm:grid-cols-2',
          columns === 3 && 'grid-cols-1 sm:grid-cols-3',
        )}
      >
        {options.map((option) => (
          <label
            key={option.value}
            className="flex cursor-pointer items-start gap-3 rounded-[var(--radius-control)] border-2 border-line-strong bg-surface p-4 has-[:checked]:border-brand has-[:checked]:bg-brand-soft has-[:focus-visible]:outline has-[:focus-visible]:outline-3 has-[:focus-visible]:outline-brand"
          >
            <input
              type="radio"
              name={name}
              value={option.value}
              defaultChecked={defaultValue === option.value}
              className="mt-1 h-5 w-5 accent-[var(--color-brand)]"
            />
            <span>
              <span className="block font-semibold">{option.label}</span>
              {option.description ? (
                <span className="block text-sm text-ink-muted">{option.description}</span>
              ) : null}
            </span>
          </label>
        ))}
      </div>
    </fieldset>
  );
}

/** Native checkbox, for the same submit-before-hydration reason as RadioCards. */
export function Checkbox({
  name,
  label,
  description,
  defaultChecked,
  value,
}: {
  name: string;
  label: string;
  description?: string;
  defaultChecked?: boolean;
  value?: string;
}) {
  return (
    <label className="mb-4 flex cursor-pointer items-start gap-3">
      <input
        type="checkbox"
        name={name}
        value={value}
        defaultChecked={defaultChecked}
        className="mt-1 h-5 w-5 accent-[var(--color-brand)]"
      />
      <span>
        <span className="block font-semibold">{label}</span>
        {description ? <span className="block text-sm text-ink-muted">{description}</span> : null}
      </span>
    </label>
  );
}

/** Live region for form-level success and failure messages. */
export function FormStatus({ status }: { status: { ok: boolean; message: string } | null }) {
  return (
    <div aria-live="polite" className="min-h-0">
      {status ? (
        <p
          className={cn(
            'mb-4 flex items-center gap-2 rounded-[var(--radius-control)] border p-3 text-sm font-medium',
            status.ok
              ? 'border-positive/40 bg-positive-soft text-positive'
              : 'border-critical/40 bg-critical-soft text-critical',
          )}
        >
          <Icon name={status.ok ? 'ok' : 'caution'} className="shrink-0" />
          {status.message}
        </p>
      ) : null}
    </div>
  );
}
