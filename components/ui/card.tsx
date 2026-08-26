import * as React from 'react';
import { Slot } from 'radix-ui';

import { cn } from '@/lib/utils';

function Card({
  className,
  asChild = false,
  ...props
}: React.ComponentProps<'div'> & { asChild?: boolean }) {
  // `asChild` is a DiaLog addition. A card is often a list item or an
  // <article>, and a <div> wrapper around an <li> breaks the list semantics a
  // screen reader relies on to announce "3 of 12".
  const Comp = asChild ? Slot.Root : 'div';
  return (
    <Comp
      data-slot="card"
      className={cn(
        // No drop shadow: DiaLog separates surfaces with a hairline border, so a
        // page of cards reads as a document rather than as floating panels.
        'flex flex-col gap-5 rounded-[var(--radius-card)] border border-line bg-card py-5 text-card-foreground sm:py-6',
        className,
      )}
      {...props}
    />
  );
}

function CardHeader({ className, ...props }: React.ComponentProps<'div'>) {
  return (
    <div
      data-slot="card-header"
      className={cn(
        '@container/card-header grid auto-rows-min grid-rows-[auto_auto] items-start gap-1.5 px-5 has-data-[slot=card-action]:grid-cols-[1fr_auto] sm:px-6 [.border-b]:pb-5',
        className,
      )}
      {...props}
    />
  );
}

function CardTitle({
  className,
  asChild = false,
  ...props
}: React.ComponentProps<'div'> & { asChild?: boolean }) {
  // `asChild` is a DiaLog addition, and it matters more than it looks. Stock
  // CardTitle renders a <div>, so a page of cards has no heading structure at
  // all — nothing for a screen reader's heading list, and nothing for
  // aria-labelledby to point at. Call sites pass a real h2/h3 through this.
  const Comp = asChild ? Slot.Root : 'div';
  return (
    <Comp
      data-slot="card-title"
      className={cn('text-lg font-semibold tracking-tight sm:text-xl', className)}
      {...props}
    />
  );
}

function CardDescription({ className, ...props }: React.ComponentProps<'div'>) {
  return (
    <div
      data-slot="card-description"
      className={cn('dl-measure text-sm text-muted-foreground', className)}
      {...props}
    />
  );
}

function CardAction({ className, ...props }: React.ComponentProps<'div'>) {
  return (
    <div
      data-slot="card-action"
      className={cn('col-start-2 row-span-2 row-start-1 self-start justify-self-end', className)}
      {...props}
    />
  );
}

function CardContent({ className, ...props }: React.ComponentProps<'div'>) {
  return <div data-slot="card-content" className={cn('px-5 sm:px-6', className)} {...props} />;
}

function CardFooter({ className, ...props }: React.ComponentProps<'div'>) {
  return (
    <div
      data-slot="card-footer"
      className={cn('flex items-center px-5 sm:px-6 [.border-t]:pt-5', className)}
      {...props}
    />
  );
}

export { Card, CardHeader, CardFooter, CardTitle, CardAction, CardDescription, CardContent };
