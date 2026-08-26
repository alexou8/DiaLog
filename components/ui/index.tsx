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
import { Icon, isIconName, type IconName } from './icon';

/**
 * Icon props accept either a registry name or a ready-made node. Call sites
 * pass a name; the registry keeps one glyph per concept across the product.
 */
export type IconProp = IconName | ReactNode;

function renderIcon(icon: IconProp, className?: string) {
  if (icon == null || icon === false) return null;
  if (isIconName(icon)) return <Icon name={icon} className={className} />;
  return icon;
}

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
        <Heading id={id} className="text-lg font-semibold tracking-tight sm:text-xl">
          {title}
        </Heading>
        {description ? (
          <p className="dl-measure mt-1 text-sm text-ink-muted">{description}</p>
        ) : null}
      </div>
      {action}
    </div>
  );
}

/**
 * A page section that is *not* a card. Most content does not need a surface:
 * a heading and a rule give it hierarchy without the "every fact in its own
 * floating rectangle" look. Reach for `Card` only when the content genuinely
 * needs to be separated from what surrounds it.
 */
export function Section({
  title,
  description,
  action,
  children,
  level = 2,
  id,
  className,
}: {
  title: ReactNode;
  description?: ReactNode;
  action?: ReactNode;
  children: ReactNode;
  level?: 2 | 3;
  id?: string;
  className?: string;
}) {
  return (
    <section className={className} aria-labelledby={id}>
      <CardHeader title={title} description={description} action={action} level={level} id={id} />
      {children}
    </section>
  );
}

export { Icon } from './icon';
export type { IconName } from './icon';

const BUTTON_BASE =
  'inline-flex items-center justify-center gap-2 rounded-[var(--radius-control)] px-5 py-2.5 text-base font-semibold transition-colors disabled:cursor-not-allowed disabled:opacity-60';

const BUTTON_VARIANTS = {
  primary: 'bg-brand text-white hover:bg-brand-strong dark:text-[oklch(19%_0.012_250)]',
  secondary:
    'border border-line-strong bg-surface text-ink hover:border-brand hover:text-brand-ink',
  ghost: 'text-brand-ink hover:bg-brand-soft',
  danger: 'border border-critical text-critical hover:bg-critical-soft',
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
  icon?: IconProp;
}) {
  return (
    <span
      className={clsx(
        'inline-flex items-center gap-1.5 rounded-[var(--radius-control)] border px-2 py-0.5 text-sm font-medium',
        TONE_SOFT[tone],
      )}
    >
      {renderIcon(icon, 'shrink-0')}
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
  icon?: IconProp;
}) {
  return (
    <div
      className={clsx('flex gap-3 rounded-[var(--radius-control)] border p-4', TONE_SOFT[tone])}
      role="note"
    >
      {icon ? <span className="mt-0.5 shrink-0">{renderIcon(icon)}</span> : null}
      <div className="min-w-0">
        {title ? <p className="font-semibold">{title}</p> : null}
        <div className={clsx('dl-measure text-sm', title && 'mt-1')}>{children}</div>
      </div>
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
  icon = 'info',
}: {
  title: string;
  children: ReactNode;
  action?: ReactNode;
  icon?: IconProp;
}) {
  return (
    <div className="rounded-[var(--radius-card)] border border-dashed border-line-strong bg-surface-sunken px-6 py-10 text-center">
      <span className="inline-flex h-10 w-10 items-center justify-center rounded-full border border-line bg-surface text-ink-muted">
        {renderIcon(icon)}
      </span>
      <p className="mt-4 text-lg font-semibold">{title}</p>
      <div className="dl-measure mx-auto mt-2 text-ink-muted">{children}</div>
      {action ? <div className="mt-6 flex flex-wrap justify-center gap-3">{action}</div> : null}
    </div>
  );
}

export function ErrorState({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div
      role="alert"
      className="rounded-[var(--radius-card)] border border-critical/40 bg-critical-soft p-6"
    >
      <p className="flex items-center gap-2 font-semibold text-critical">
        <Icon name="alert" className="shrink-0" />
        {title}
      </p>
      {children ? <div className="dl-measure mt-2 text-sm">{children}</div> : null}
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
    <header className="mb-8 flex flex-wrap items-end justify-between gap-4 border-b border-line pb-5">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight sm:text-[1.75rem]">{title}</h1>
        {description ? <p className="dl-measure mt-2 text-ink-muted">{description}</p> : null}
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
    <details className="mt-3 rounded-[var(--radius-control)] border border-line bg-surface-sunken">
      <summary className="flex cursor-pointer list-none items-center gap-2 px-4 py-3 text-sm font-semibold text-brand-ink marker:content-none">
        <Icon name="info" className="shrink-0" />
        {label}
      </summary>
      <div className="dl-measure border-t border-line px-4 py-3 text-sm text-ink-muted">
        {children}
      </div>
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
    <div className={clsx('rounded-[var(--radius-control)] border p-4', TONE_SOFT[tone])}>
      <p className="text-sm font-medium">{label}</p>
      <p className="dl-numeric mt-1 text-[1.75rem] font-semibold leading-none sm:text-3xl">
        {value}
        {unit ? <span className="ml-1.5 text-base font-medium tracking-normal">{unit}</span> : null}
      </p>
      {hint ? <p className="mt-2 text-sm opacity-90">{hint}</p> : null}
    </div>
  );
}

export const MEDICAL_DISCLAIMER =
  'DiaLog organises and explains your own data. It is not a medical device and does not provide medical advice, diagnosis or treatment. Always talk to your healthcare professional before changing anything about your care.';

export function MedicalDisclaimer({ compact = false }: { compact?: boolean }) {
  return (
    <p className={clsx('dl-measure flex gap-2 text-ink-muted', compact ? 'text-xs' : 'text-sm')}>
      <Icon name="info" className="mt-0.5 shrink-0" />
      <span>{MEDICAL_DISCLAIMER}</span>
    </p>
  );
}
