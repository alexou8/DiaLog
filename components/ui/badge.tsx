/**
 * Badge — shadcn/ui primitive carrying DiaLog's status tones.
 *
 * The variants are health-status tones, not visual weights, and every one is
 * a soft fill plus a matching border and text colour drawn from the tokens in
 * globals.css. Stock shadcn badges are 12px text; these are 14px, because a
 * badge here is often the label for a glucose classification and is read, not
 * decoration.
 *
 * A tone is never the only signal. Call sites pass an icon and a word ("Above
 * your range"), so the meaning survives greyscale and a screen reader.
 */
import * as React from 'react';
import { Slot } from 'radix-ui';

import { cn } from '@/lib/utils';
import { toneVariants, type Tone } from './tone';

const BADGE_BASE =
  'inline-flex w-fit shrink-0 items-center justify-center gap-1.5 rounded-[var(--radius-control)] border px-2 py-0.5 text-sm font-medium whitespace-nowrap [&>svg]:pointer-events-none [&>svg]:size-[1.05em]';

/** Exported so a link or a <summary> can be styled as a badge via `cn`. */
export const badgeVariants = ({ tone }: { tone?: Tone } = {}) =>
  cn(BADGE_BASE, toneVariants({ tone }));

export type BadgeTone = Tone;

function Badge({
  className,
  tone,
  asChild = false,
  ...props
}: React.ComponentProps<'span'> & { tone?: Tone; asChild?: boolean }) {
  const Comp = asChild ? Slot.Root : 'span';
  return <Comp data-slot="badge" className={cn(badgeVariants({ tone }), className)} {...props} />;
}

export { Badge };
