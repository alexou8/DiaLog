/**
 * Accessible form primitives.
 *
 * Every field is a real label bound to a real control. Errors are associated
 * via aria-describedby and announced through a live region, and they are
 * written in plain language ("Please enter a number between 1.1 and 38.9")
 * rather than as validation jargon.
 */
'use client';

import { useId, type ReactNode } from 'react';
import clsx from 'clsx';

const CONTROL =
  'w-full rounded-xl border-2 border-line-strong bg-surface px-4 py-3 text-base text-ink placeholder:text-ink-muted/70 focus:border-brand';

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
      <label htmlFor={id} className="mb-1.5 block text-base font-semibold">
        {label}
        {required ? (
          <span className="ml-1 font-normal text-ink-muted">(required)</span>
        ) : (
          <span className="ml-1 font-normal text-ink-muted">(optional)</span>
        )}
      </label>
      {hint ? (
        <p id={hintId} className="mb-2 text-sm text-ink-muted">
          {hint}
        </p>
      ) : null}
      {children({ id, describedBy, invalid: Boolean(error) })}
      {error ? (
        <p id={errorId} className="mt-1.5 text-sm font-medium text-critical">
          <span aria-hidden="true">⚠ </span>
          {error}
        </p>
      ) : null}
    </div>
  );
}

export function TextInput({
  invalid,
  className,
  ...rest
}: React.InputHTMLAttributes<HTMLInputElement> & { invalid?: boolean }) {
  return (
    <input
      {...rest}
      aria-invalid={invalid || undefined}
      className={clsx(CONTROL, invalid && 'border-critical', className)}
    />
  );
}

export function TextArea({
  invalid,
  className,
  ...rest
}: React.TextareaHTMLAttributes<HTMLTextAreaElement> & { invalid?: boolean }) {
  return (
    <textarea
      {...rest}
      aria-invalid={invalid || undefined}
      className={clsx(CONTROL, 'min-h-28', invalid && 'border-critical', className)}
    />
  );
}

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
      className={clsx(CONTROL, invalid && 'border-critical', className)}
    >
      {children}
    </select>
  );
}

/**
 * Large tappable radio group — used instead of dropdowns wherever there are
 * few options, because a visible set of big targets is easier than a picker
 * for users with limited dexterity or eyesight.
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
  return (
    <fieldset className="mb-5">
      <legend className="mb-1.5 text-base font-semibold">{legend}</legend>
      {hint ? <p className="mb-2 text-sm text-ink-muted">{hint}</p> : null}
      <div
        className={clsx(
          'grid gap-2',
          columns === 1 && 'grid-cols-1',
          columns === 2 && 'grid-cols-1 sm:grid-cols-2',
          columns === 3 && 'grid-cols-1 sm:grid-cols-3',
        )}
      >
        {options.map((option) => (
          <label
            key={option.value}
            className="flex cursor-pointer items-start gap-3 rounded-xl border-2 border-line-strong bg-surface p-4 has-[:checked]:border-brand has-[:checked]:bg-brand-soft has-[:focus-visible]:outline has-[:focus-visible]:outline-3 has-[:focus-visible]:outline-brand"
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

export function Checkbox({
  name,
  label,
  description,
  defaultChecked,
}: {
  name: string;
  label: string;
  description?: string;
  defaultChecked?: boolean;
}) {
  return (
    <label className="mb-4 flex cursor-pointer items-start gap-3">
      <input
        type="checkbox"
        name={name}
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
          className={clsx(
            'mb-4 rounded-xl border p-3 text-sm font-medium',
            status.ok
              ? 'border-positive/40 bg-positive-soft text-positive'
              : 'border-critical/40 bg-critical-soft text-critical',
          )}
        >
          <span aria-hidden="true">{status.ok ? '✓ ' : '⚠ '}</span>
          {status.message}
        </p>
      ) : null}
    </div>
  );
}
