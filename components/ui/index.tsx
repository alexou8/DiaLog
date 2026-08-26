/**
 * DiaLog's semantic layer over shadcn/ui.
 *
 * shadcn is the foundation: every primitive here renders a shadcn component
 * (which in turn renders a Radix primitive where one exists), retuned to the
 * DiaLog design tokens in `globals.css`. What this file adds on top is
 * *meaning*: a call site asks for a `Callout` with tone "caution", not for an
 * Alert with a set of utility classes. That is what keeps twelve callouts
 * across nine pages looking like one product.
 *
 * The rules the primitives below encode, so no call site has to remember them:
 *
 * - Real semantics. A real <button>, a real <label>, a real <table>, and list
 *   items that stay list items.
 * - A 44px minimum touch target and 17px body text everywhere, including on
 *   desktop. Stock shadcn sizes are smaller on both counts; see button.tsx.
 * - One focus treatment, defined once in globals.css and never removed.
 * - Colour is never the only signal. Every tone is paired with a text label,
 *   and usually an icon, at the call site.
 */
import type { ReactNode } from 'react';
import Link from 'next/link';

import { cn } from '@/lib/utils';
import { Icon, isIconName, type IconName } from './icon';
import { Button as ShadButton, type ButtonVariant } from './button';
import { Badge as ShadBadge } from './badge';
import { toneVariants, type Tone } from './tone';
import { Alert, AlertDescription, AlertTitle } from './alert';
import {
  Card as ShadCard,
  CardAction,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader as ShadCardHeader,
  CardTitle,
} from './card';
import { Skeleton } from './skeleton';

/** Health status tones. Shared by Badge, Alert, Callout and Stat. */
export type { Tone };

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

/* -------------------------------------------------------------- surfaces */

/**
 * A padded surface. Wraps shadcn's Card, but flattens its column layout so
 * children keep their own vertical rhythm, and adds the horizontal padding
 * shadcn leaves to CardContent — because most DiaLog cards hold one block of
 * content rather than the header/content/footer trio.
 *
 * Reach for `Section` first. Not everything needs a surface.
 */
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
    <ShadCard asChild className={cn('gap-0 px-5 sm:px-6', className)}>
      <As>{children}</As>
    </ShadCard>
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
    // px-0 because the surrounding Card already pads; shadcn's CardHeader
    // assumes it is the one applying it.
    <ShadCardHeader className="mb-4 px-0">
      <CardTitle asChild>
        <Heading id={id}>{title}</Heading>
      </CardTitle>
      {description ? <CardDescription>{description}</CardDescription> : null}
      {action ? <CardAction>{action}</CardAction> : null}
    </ShadCardHeader>
  );
}

/**
 * A page section that is *not* a card. Most content does not need a surface:
 * a heading and a rule give it hierarchy without the "every fact in its own
 * floating rectangle" look.
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

/* --------------------------------------------------------------- actions */

export { ShadButton as Button };
export type { ButtonVariant };

/**
 * A link that looks and behaves like a button. `asChild` hands the button's
 * styling to a real <a>, so it keeps link semantics: opening in a new tab,
 * "copy link address", and the correct screen-reader role all still work.
 */
export function ButtonLink({
  children,
  href,
  variant,
  size,
  className,
  ...rest
}: {
  children: ReactNode;
  href: string;
  variant?: ButtonVariant;
  size?: React.ComponentProps<typeof ShadButton>['size'];
  className?: string;
} & Omit<React.ComponentProps<typeof Link>, 'href' | 'className'>) {
  return (
    <ShadButton asChild variant={variant} size={size} className={cn('dl-target', className)}>
      <Link href={href} {...rest}>
        {children}
      </Link>
    </ShadButton>
  );
}

/* ------------------------------------------------------- status and tone */

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
    <ShadBadge tone={tone}>
      {renderIcon(icon, 'shrink-0')}
      {children}
    </ShadBadge>
  );
}

/**
 * A contextual note. `role="note"` rather than `role="alert"`: these explain
 * something on a page the user chose to open, and should not interrupt a
 * screen reader mid-sentence. `ErrorState` is the one that announces.
 */
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
    <Alert tone={tone}>
      {renderIcon(icon)}
      {title ? <AlertTitle>{title}</AlertTitle> : null}
      <AlertDescription>{children}</AlertDescription>
    </Alert>
  );
}

export function ErrorState({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <Alert tone="critical" role="alert" className="p-6">
      <Icon name="alert" />
      <AlertTitle>{title}</AlertTitle>
      {children ? <AlertDescription>{children}</AlertDescription> : null}
    </Alert>
  );
}

/**
 * Empty states teach the next action instead of showing an empty chart. They
 * say what the screen is, why it is empty, and what to do about it.
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
 * "Why am I seeing this?" disclosure.
 *
 * Deliberately a native <details> rather than shadcn's Accordion. Accordion is
 * a client component, and this sits inside server-rendered pages that would
 * otherwise stay on the server; more importantly, <details> opens with
 * JavaScript unavailable or still loading, which is the state a slow phone on
 * mobile data is actually in. The evidence behind an observation is not
 * something to hide behind a hydration boundary.
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

/** One health measurement. Tabular figures so a row of these stays aligned. */
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
    // A tinted panel, not an Alert. A dashboard row of six metrics announcing
    // themselves as notes is noise; the tone here is colour, and the meaning
    // is in the label and the hint beside it.
    <div className={cn('rounded-[var(--radius-control)] border p-4', toneVariants({ tone }))}>
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
    <p className={cn('dl-measure flex gap-2 text-ink-muted', compact ? 'text-xs' : 'text-sm')}>
      <Icon name="info" className="mt-0.5 shrink-0" />
      <span>{MEDICAL_DISCLAIMER}</span>
    </p>
  );
}

/* ------------------------------------------- shadcn primitives, re-exported */

export { Skeleton };
export { CardContent, CardDescription, CardFooter, CardTitle, CardAction };
export { Alert, AlertTitle, AlertDescription };
export { Icon } from './icon';
export type { IconName } from './icon';
