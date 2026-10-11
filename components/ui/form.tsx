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
 * - Errors are associated with `aria-describedby`, marked with `aria-invalid`,
 *   collected into one summary that takes focus, and written in plain
 *   language ("Please enter a number between 1.1 and 38.9") rather than as
 *   validation jargon.
 * - Hints sit above the control, not in a tooltip. Nothing a person needs in
 *   order to answer correctly is hidden behind a hover.
 * - A failed submit never throws away what the person typed. See `ActionForm`.
 */
'use client';

import {
  createContext,
  startTransition,
  useContext,
  useEffect,
  useId,
  useRef,
  type FormEvent,
  type MouseEvent,
  type ReactNode,
  type RefObject,
} from 'react';
import { useFormStatus } from 'react-dom';

import { cn } from '@/lib/utils';
import { Button } from './button';
import { Icon } from './icon';
import { Input } from './input';
import { Label } from './label';
import { Textarea } from './textarea';

/**
 * The shape every form-backing Server Action returns. `values` is the
 * submitted form, echoed back on failure (see `ActionForm` for why). It is
 * structurally compatible with the older `{ ok, message, errors }` states, so
 * an action that does not echo values still works here, it just cannot
 * restore a form after a no-JavaScript round trip.
 */
export interface FormActionState {
  ok: boolean;
  message?: string;
  errors?: Record<string, string>;
  values?: Record<string, string | string[]>;
}

interface FormContextValue {
  state: FormActionState | null;
  pending: boolean;
  fieldId: (name: string) => string;
}

const FormContext = createContext<FormContextValue | null>(null);

/** The value this field was last submitted with, when the action echoed it. */
function submittedValue(ctx: FormContextValue | null, name: string | undefined) {
  if (!ctx || !name) return undefined;
  const value = ctx.state?.values?.[name];
  return Array.isArray(value) ? value[0] : value;
}

function contextError(ctx: FormContextValue | null, name: string | undefined) {
  if (!ctx || !name) return undefined;
  return ctx.state?.errors?.[name];
}

/**
 * The one form pattern for every form that posts to a Server Action.
 *
 * Why it exists, in the order the bugs were found:
 *
 * 1. React 19 resets every uncontrolled field once a `<form action>` resolves,
 *    and returning a validation error still counts as resolving. Typing "45"
 *    in mmol/L used to clear the value, the note and the chosen context at the
 *    same moment the error appeared. With JavaScript, `onSubmit` dispatches
 *    the action itself inside a transition, which React documents as opting
 *    out of that reset (it still drives `useFormStatus`). Without JavaScript
 *    the browser posts and the server renders the page again, so the action
 *    echoes `values` back and every control below reads its default from
 *    them.
 * 2. Field errors used to come back with no form-level message, and every
 *    form gated its live region on that message, so a screen-reader user
 *    heard nothing at all. The summary below is rendered for any failure,
 *    lists each problem as a link to its field, and takes focus.
 * 3. Live regions are scoped to the status line only. Wrapping a whole form
 *    or result in `aria-live` re-reads every editable field on each change.
 */
export function ActionForm({
  action,
  state,
  pending = false,
  labels,
  children,
  className,
  formRef,
  showSuccess = true,
  shouldFocusSummary,
  ...rest
}: {
  /** The dispatch function returned by `useActionState`. */
  action: (formData: FormData) => void;
  state: FormActionState | null;
  /** `useActionState`'s third value; also drives `SubmitButton`. */
  pending?: boolean;
  /** Field name → visible label, used for the error summary's links. */
  labels?: Record<string, string>;
  children: ReactNode;
  className?: string;
  formRef?: RefObject<HTMLFormElement | null>;
  /** Show `state.message` as a status line when the action succeeds. */
  showSuccess?: boolean;
  /**
   * Asked before the error summary takes focus after a failed submit. For a
   * page that submits several forms at once and then decides where focus goes
   * itself (quick-log's "Save all"); otherwise each failing form would pull
   * focus in turn. Pass a stable function (useCallback): it is an effect
   * dependency.
   */
  shouldFocusSummary?: () => boolean;
  'aria-labelledby'?: string;
  'aria-label'?: string;
}) {
  const prefix = `f${useId().replace(/[^\w-]/g, '')}`;
  const summaryRef = useRef<HTMLDivElement>(null);

  // Focus moves to the summary after a failed submit, so the first thing a
  // keyboard or screen-reader user meets is what went wrong and a link to
  // each field. Keyed on the state object, which is new for every response.
  useEffect(() => {
    if (state && !state.ok && (shouldFocusSummary?.() ?? true)) summaryRef.current?.focus();
  }, [state, shouldFocusSummary]);

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const submitter = (event.nativeEvent as SubmitEvent).submitter;
    const formData = new FormData(event.currentTarget, submitter);
    startTransition(() => action(formData));
  }

  const context: FormContextValue = {
    state,
    pending,
    fieldId: (name: string) => `${prefix}-${name}`,
  };

  return (
    <FormContext.Provider value={context}>
      <form
        ref={formRef}
        action={action}
        onSubmit={handleSubmit}
        noValidate
        className={className}
        {...rest}
      >
        <FormErrorSummary
          state={state}
          labels={labels}
          fieldId={context.fieldId}
          summaryRef={summaryRef}
        />
        <FormStatus
          status={
            showSuccess && state?.ok && state.message ? { ok: true, message: state.message } : null
          }
        />
        {children}
      </form>
    </FormContext.Provider>
  );
}

/**
 * Form-level error summary. `role="alert"` on the summary alone, rendered only
 * after a failed submit, so it is announced once when it appears and never
 * re-reads the fields themselves.
 */
function FormErrorSummary({
  state,
  labels,
  fieldId,
  summaryRef,
}: {
  state: FormActionState | null;
  labels?: Record<string, string>;
  fieldId: (name: string) => string;
  summaryRef: RefObject<HTMLDivElement | null>;
}) {
  const titleId = useId();
  if (!state || state.ok) return null;
  const problems = Object.entries(state.errors ?? {});
  if (problems.length === 0 && !state.message) return null;

  const title =
    problems.length === 0
      ? state.message
      : problems.length === 1
        ? 'Please check one answer'
        : `Please check ${problems.length} answers`;

  function focusField(event: MouseEvent<HTMLAnchorElement>, name: string) {
    const target = document.getElementById(fieldId(name));
    if (!target) return;
    // A fragment link scrolls to the field but does not reliably focus it;
    // a person following "Your reading: please enter a number" expects to be
    // able to type straight away.
    event.preventDefault();
    target.focus();
    target.scrollIntoView({ block: 'center' });
  }

  return (
    <div
      ref={summaryRef}
      tabIndex={-1}
      role="alert"
      aria-labelledby={titleId}
      className="mb-5 rounded-[var(--radius-control)] border-2 border-critical bg-critical-soft p-4 text-ink"
    >
      <p id={titleId} className="flex items-start gap-2 font-semibold text-critical">
        <Icon name="alert" className="mt-1 shrink-0" />
        <span>{title}</span>
      </p>
      {problems.length > 0 && state.message ? <p className="mt-2">{state.message}</p> : null}
      {problems.length > 0 ? (
        <ul className="mt-2 flex list-disc flex-col gap-1 pl-10">
          {problems.map(([name, message]) => {
            const label = labels?.[name];
            return (
              <li key={name}>
                {label ? (
                  <a
                    href={`#${fieldId(name)}`}
                    onClick={(event) => focusField(event, name)}
                    className="font-medium underline underline-offset-4"
                  >
                    {label}: {message}
                  </a>
                ) : (
                  message
                )}
              </li>
            );
          })}
        </ul>
      ) : null}
    </div>
  );
}

/**
 * Submit button that shows a pending label. Reads the pending flag from the
 * surrounding `ActionForm` and from `useFormStatus`, so it works in either.
 */
export function SubmitButton({
  children,
  pendingLabel = 'Saving…',
  className,
  disabled,
  ...rest
}: Omit<React.ComponentProps<typeof Button>, 'type'> & { pendingLabel?: string }) {
  const ctx = useContext(FormContext);
  const status = useFormStatus();
  const pending = Boolean(ctx?.pending) || status.pending;
  return (
    <Button type="submit" className={className} disabled={pending || disabled} {...rest}>
      {pending ? pendingLabel : children}
    </Button>
  );
}

export function Field({
  label,
  hint,
  error,
  children,
  required,
  htmlFor,
  name,
}: {
  label: string;
  hint?: ReactNode;
  /** Overrides the error read from the surrounding `ActionForm` by `name`. */
  error?: string;
  children: (ids: { id: string; describedBy: string | undefined; invalid: boolean }) => ReactNode;
  required?: boolean;
  htmlFor?: string;
  /**
   * The control's form name. Inside an `ActionForm` this gives the control a
   * stable id the error summary can link to, and picks up its error.
   */
  name?: string;
}) {
  const ctx = useContext(FormContext);
  const generated = useId();
  const id = htmlFor ?? (ctx && name ? ctx.fieldId(name) : generated);
  const resolvedError = error ?? contextError(ctx, name);
  const hintId = hint ? `${id}-hint` : undefined;
  const errorId = resolvedError ? `${id}-error` : undefined;
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
      {children({ id, describedBy, invalid: Boolean(resolvedError) })}
      {resolvedError ? <FieldError id={errorId}>{resolvedError}</FieldError> : null}
    </div>
  );
}

function FieldError({ id, children }: { id?: string; children: ReactNode }) {
  return (
    <p id={id} className="mt-1.5 flex items-start gap-1.5 text-sm font-medium text-critical">
      <Icon name="caution" className="mt-0.5 shrink-0" />
      <span>{children}</span>
    </p>
  );
}

/**
 * Restores a submitted value after a no-JavaScript round trip, unless the
 * control is controlled or is a password (never echoed, never re-rendered).
 */
function useRestoredDefault<T>(
  name: string | undefined,
  defaultValue: T,
  controlled: boolean,
  type?: string,
): T | string {
  const ctx = useContext(FormContext);
  if (controlled || type === 'password' || type === 'file') return defaultValue;
  return submittedValue(ctx, name) ?? defaultValue;
}

function useContextInvalid(name: string | undefined, invalid: boolean | undefined) {
  const ctx = useContext(FormContext);
  return invalid || Boolean(contextError(ctx, name)) || undefined;
}

export function TextInput({
  invalid,
  name,
  defaultValue,
  ...rest
}: React.ComponentProps<typeof Input> & { invalid?: boolean }) {
  const restored = useRestoredDefault(name, defaultValue, rest.value !== undefined, rest.type);
  const ariaInvalid = useContextInvalid(name, invalid);
  return (
    <Input
      {...rest}
      name={name}
      defaultValue={restored as typeof defaultValue}
      aria-invalid={ariaInvalid}
    />
  );
}

export function TextArea({
  invalid,
  name,
  defaultValue,
  ...rest
}: React.ComponentProps<typeof Textarea> & { invalid?: boolean }) {
  const restored = useRestoredDefault(name, defaultValue, rest.value !== undefined);
  const ariaInvalid = useContextInvalid(name, invalid);
  return (
    <Textarea
      {...rest}
      name={name}
      defaultValue={restored as typeof defaultValue}
      aria-invalid={ariaInvalid}
    />
  );
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
 * The same reasoning applies to RadioCards and Checkbox below. The one Radix
 * Select in the product is the device-level theme picker (ThemeToggle), which
 * never posts a form; nothing whose value has to survive a submit that beats
 * hydration uses a Radix control.
 */
export function Select({
  invalid,
  className,
  children,
  name,
  defaultValue,
  ...rest
}: React.SelectHTMLAttributes<HTMLSelectElement> & { invalid?: boolean }) {
  const restored = useRestoredDefault(name, defaultValue, rest.value !== undefined);
  const ariaInvalid = useContextInvalid(name, invalid);
  return (
    <select
      {...rest}
      name={name}
      defaultValue={restored as typeof defaultValue}
      aria-invalid={ariaInvalid}
      className={cn(
        'min-h-11 w-full rounded-[var(--radius-control)] border-2 border-input bg-surface px-4 py-2.5 text-base text-ink transition-colors focus:border-brand aria-invalid:border-destructive aria-invalid:border-l-[6px]',
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
  error,
}: {
  name: string;
  legend: string;
  hint?: string;
  options: { value: T; label: string; description?: string }[];
  defaultValue?: T;
  columns?: 1 | 2 | 3;
  error?: string;
}) {
  const ctx = useContext(FormContext);
  const id = useId();
  const hintId = hint ? `${id}-hint` : undefined;
  const resolvedError = error ?? contextError(ctx, name);
  const errorId = resolvedError ? `${id}-error` : undefined;
  const describedBy = [hintId, errorId].filter(Boolean).join(' ') || undefined;
  const selected = submittedValue(ctx, name) ?? defaultValue;

  return (
    // The error is tied to the group through the fieldset's description, not
    // `aria-invalid` on each radio: that attribute is not supported on the
    // radio role, and the group is what is wrong, not any one option.
    <fieldset className="mb-5" aria-describedby={describedBy}>
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
        {options.map((option, index) => (
          <label
            key={option.value}
            className={cn(
              'flex cursor-pointer items-start gap-3 rounded-[var(--radius-control)] border-2 border-line-strong bg-surface p-4 has-[:checked]:border-brand has-[:checked]:bg-brand-soft has-[:focus-visible]:outline has-[:focus-visible]:outline-3 has-[:focus-visible]:outline-brand',
              resolvedError && 'border-critical',
            )}
          >
            <input
              type="radio"
              name={name}
              value={option.value}
              // The error summary links here, to the first option.
              id={index === 0 && ctx ? ctx.fieldId(name) : undefined}
              defaultChecked={selected === option.value}
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
      {resolvedError ? <FieldError id={errorId}>{resolvedError}</FieldError> : null}
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
  const ctx = useContext(FormContext);
  const values = ctx?.state?.values;
  // After an echoed failure, a box that is missing from the submission was
  // unchecked on purpose; otherwise fall back to the caller's default.
  let checked = defaultChecked;
  if (values) {
    const submitted = values[name];
    const own = value ?? 'on';
    checked = Array.isArray(submitted) ? submitted.includes(own) : submitted === own;
  }
  return (
    <label className="mb-4 flex cursor-pointer items-start gap-3">
      <input
        type="checkbox"
        name={name}
        value={value}
        defaultChecked={checked}
        className="mt-1 h-5 w-5 accent-[var(--color-brand)]"
      />
      <span>
        <span className="block font-semibold">{label}</span>
        {description ? <span className="block text-sm text-ink-muted">{description}</span> : null}
      </span>
    </label>
  );
}

/**
 * Live region for form-level status messages. The region itself is always in
 * the DOM (a live region inserted together with its text is often not
 * announced), and only the message inside it changes.
 */
export function FormStatus({ status }: { status: { ok: boolean; message: string } | null }) {
  return (
    <div role="status" aria-live="polite" className="min-h-0">
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
