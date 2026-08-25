/**
 * DiaLog UI primitives.
 *
 * Deliberately small and semantic: real <button>, real <label>, real <table>.
 * Every interactive element has a visible focus ring (see globals.css) and a
 * 44px minimum touch target. Nothing here relies on colour alone to convey
 * state.
 */
import type { ReactNode } from 'react';
import Link from 'next/link';
import clsx from 'clsx';

type Tone = 'brand' | 'positive' | 'notice' | 'caution' | 'critical' | 'info' | 'neutral';

const TONE_SOFT: Record<Tone, string> = {
  brand: 'bg-brand-soft text-brand-ink border-brand/30',
  positive: 'bg-positive-soft text-positive border-positive/30',
  notice: 'bg-notice-soft text-notice border-notice/30',
  caution: 'bg-caution-soft text-caution border-caution/30',
  critical: 'bg-critical-soft text-critical border-critical/30',
  info: 'bg-info-soft text-info border-info/30',
  neutral: 'bg-surface-sunken text-ink-muted border-line',
};

export function Card({
  children,
  className,
  as: As = 'section',
}: {
  children: ReactNode;
  className?: string;
  as?: 'section' | 'div' | 'article' | 'li';
}) {
  return (
    <As
      className={clsx(
        'rounded-[var(--radius-card)] border border-line bg-surface p-5 sm:p-6',
        className,
      )}
    >
      {children}
    </As>
  );
}

export function CardHeader({
  title,
  description,
  action,
  level = 2,
  id,
}: {
  title: ReactNode;
  description?: ReactNode;
  action?: ReactNode;
  level?: 2 | 3;
  id?: string;
}) {
  const Heading = level === 2 ? 'h2' : 'h3';
  return (
    <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
      <div>
        <Heading id={id} className="text-lg font-semibold sm:text-xl">
          {title}
        </Heading>
        {description ? <p className="mt-1 text-sm text-ink-muted">{description}</p> : null}
      </div>
      {action}
    </div>
  );
}

const BUTTON_BASE =
  'inline-flex items-center justify-center gap-2 rounded-xl px-5 py-2.5 text-base font-semibold transition-colors disabled:cursor-not-allowed disabled:opacity-60';

const BUTTON_VARIANTS = {
  primary: 'bg-brand text-white hover:bg-brand-strong dark:text-[oklch(19%_0.012_250)]',
  secondary:
    'border-2 border-line-strong bg-surface text-ink hover:border-brand hover:text-brand-ink',
  ghost: 'text-brand-ink hover:bg-brand-soft',
  danger: 'border-2 border-critical text-critical hover:bg-critical-soft',
} as const;

export type ButtonVariant = keyof typeof BUTTON_VARIANTS;

export function Button({
  children,
  variant = 'primary',
  className,
  type = 'button',
  ...rest
}: React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: ButtonVariant }) {
  return (
    <button
      type={type}
      className={clsx(BUTTON_BASE, BUTTON_VARIANTS[variant], className)}
      {...rest}
    >
      {children}
    </button>
  );
}

export function ButtonLink({
  children,
  href,
  variant = 'primary',
  className,
  ...rest
}: {
  children: ReactNode;
  href: string;
  variant?: ButtonVariant;
  className?: string;
} & Omit<React.ComponentProps<typeof Link>, 'href' | 'className'>) {
  return (
    <Link
      href={href}
      className={clsx(BUTTON_BASE, 'dl-target', BUTTON_VARIANTS[variant], className)}
      {...rest}
    >
      {children}
    </Link>
  );
}

export function Badge({
  children,
  tone = 'neutral',
  icon,
}: {
  children: ReactNode;
  tone?: Tone;
  icon?: ReactNode;
}) {
  return (
    <span
      className={clsx(
        'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-sm font-medium',
        TONE_SOFT[tone],
      )}
    >
      {icon ? (
        <span aria-hidden="true" className="text-[0.9em] leading-none">
          {icon}
        </span>
      ) : null}
      {children}
    </span>
  );
}

/** Non-alarming contextual message. `tone` never carries meaning by itself. */
export function Callout({
  children,
  tone = 'info',
  title,
  icon,
}: {
  children: ReactNode;
  tone?: Tone;
  title?: string;
  icon?: string;
}) {
  return (
    <div className={clsx('rounded-xl border p-4', TONE_SOFT[tone])} role="note">
      {title ? (
        <p className="font-semibold">
          {icon ? (
            <span aria-hidden="true" className="mr-2">
              {icon}
            </span>
          ) : null}
          {title}
        </p>
      ) : null}
      <div className={clsx('text-sm', title && 'mt-1')}>{children}</div>
    </div>
  );
}

/**
 * Empty states teach the next action instead of showing an empty chart.
 */
export function EmptyState({
  title,
  children,
  action,
  icon = '🌱',
}: {
  title: string;
  children: ReactNode;
  action?: ReactNode;
  icon?: string;
}) {
  return (
    <div className="rounded-[var(--radius-card)] border-2 border-dashed border-line bg-surface-sunken p-8 text-center">
      <p aria-hidden="true" className="text-3xl">
        {icon}
      </p>
      <p className="mt-3 text-lg font-semibold">{title}</p>
      <div className="mx-auto mt-2 max-w-prose text-ink-muted">{children}</div>
      {action ? <div className="mt-5 flex justify-center gap-3">{action}</div> : null}
    </div>
  );
}

export function ErrorState({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div
      role="alert"
      className="rounded-[var(--radius-card)] border border-critical/40 bg-critical-soft p-6"
    >
      <p className="font-semibold text-critical">
        <span aria-hidden="true">⚠ </span>
        {title}
      </p>
      {children ? <div className="mt-2 text-sm">{children}</div> : null}
    </div>
  );
}

/** Skeleton placeholder used by route-level loading.tsx files. */
export function Skeleton({ className }: { className?: string }) {
  return (
    <div
      className={clsx('animate-pulse rounded-lg bg-surface-sunken', className)}
      aria-hidden="true"
    />
  );
}

export function PageHeader({
  title,
  description,
  action,
}: {
  title: string;
  description?: ReactNode;
  action?: ReactNode;
}) {
  return (
    <header className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div>
        <h1 className="text-2xl font-bold tracking-tight sm:text-3xl">{title}</h1>
        {description ? <p className="mt-2 max-w-prose text-ink-muted">{description}</p> : null}
      </div>
      {action}
    </header>
  );
}

/**
 * "Why am I seeing this?" disclosure. Native <details> so it works without
 * JavaScript and is announced correctly by screen readers.
 */
export function WhyThis({
  children,
  label = 'Why am I seeing this?',
}: {
  children: ReactNode;
  label?: string;
}) {
  return (
    <details className="mt-3 rounded-lg border border-line bg-surface-sunken">
      <summary className="cursor-pointer list-none px-4 py-3 text-sm font-semibold text-brand-ink marker:content-none">
        <span aria-hidden="true">ⓘ </span>
        {label}
      </summary>
      <div className="border-t border-line px-4 py-3 text-sm text-ink-muted">{children}</div>
    </details>
  );
}

export function Stat({
  label,
  value,
  unit,
  hint,
  tone = 'neutral',
}: {
  label: string;
  value: ReactNode;
  unit?: string;
  hint?: ReactNode;
  tone?: Tone;
}) {
  return (
    <div className={clsx('rounded-xl border p-4', TONE_SOFT[tone])}>
      <p className="text-sm font-medium">{label}</p>
      <p className="mt-1 text-2xl font-bold tabular-nums sm:text-3xl">
        {value}
        {unit ? <span className="ml-1 text-base font-medium"> {unit}</span> : null}
      </p>
      {hint ? <p className="mt-1 text-sm opacity-90">{hint}</p> : null}
    </div>
  );
}

export const MEDICAL_DISCLAIMER =
  'DiaLog organises and explains your own data. It is not a medical device and does not provide medical advice, diagnosis or treatment. Always talk to your healthcare professional before changing anything about your care.';

export function MedicalDisclaimer({ compact = false }: { compact?: boolean }) {
  return (
    <p className={clsx('text-ink-muted', compact ? 'text-xs' : 'text-sm')}>
      <span aria-hidden="true">ℹ️ </span>
      {MEDICAL_DISCLAIMER}
    </p>
  );
}
