/**
 * Button — shadcn/ui primitive, retuned to DiaLog's design system.
 *
 * Three things differ from stock shadcn on purpose:
 *
 * 1. Size. Stock `default` is a 36px-tall, 14px-text control. DiaLog's floor
 *    is a 44px touch target and 17px body text, because the product is used
 *    one-handed on a phone and by people with limited dexterity or eyesight.
 *    There is no size below `sm`, and `sm` is still 44px tall.
 * 2. Variant names. They stay DiaLog's semantic names (primary / secondary /
 *    ghost / danger) rather than shadcn's visual ones, so a call site says
 *    what the action *is*, not what it looks like.
 * 3. Focus. The global `:focus-visible` treatment in globals.css is the single
 *    focus style across the product, so the stock ring utilities are dropped
 *    rather than competing with it.
 */
import * as React from 'react';
import { cva, type VariantProps } from 'class-variance-authority';
import { Slot } from 'radix-ui';

import { cn } from '@/lib/utils';

const buttonVariants = cva(
  "inline-flex shrink-0 items-center justify-center gap-2 rounded-[var(--radius-control)] font-semibold whitespace-nowrap transition-colors outline-none disabled:pointer-events-none disabled:cursor-not-allowed disabled:opacity-60 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-[1.15em]",
  {
    variants: {
      variant: {
        primary: 'bg-primary text-primary-foreground hover:bg-brand-strong',
        secondary:
          'border border-line-strong bg-surface text-ink hover:border-brand hover:text-brand-ink',
        ghost: 'text-brand-ink hover:bg-accent',
        danger: 'border border-destructive text-destructive hover:bg-critical-soft',
        link: 'text-brand-ink underline underline-offset-4 hover:text-ink',
      },
      size: {
        default: 'min-h-11 px-5 py-2.5 text-base',
        sm: 'min-h-11 px-3 py-2 text-sm',
        lg: 'min-h-12 px-6 py-3 text-base',
        icon: 'size-11',
      },
    },
    defaultVariants: { variant: 'primary', size: 'default' },
  },
);

export type ButtonVariant = NonNullable<VariantProps<typeof buttonVariants>['variant']>;

function Button({
  className,
  variant,
  size,
  asChild = false,
  type,
  ...props
}: React.ComponentProps<'button'> & VariantProps<typeof buttonVariants> & { asChild?: boolean }) {
  const Comp = asChild ? Slot.Root : 'button';

  return (
    <Comp
      data-slot="button"
      // A <button> inside a <form> defaults to type="submit", which has
      // submitted forms by accident here before. Explicit unless overridden,
      // and left alone when the button is really an <a> via asChild.
      type={asChild ? type : (type ?? 'button')}
      className={cn(buttonVariants({ variant, size }), className)}
      {...props}
    />
  );
}

export { Button, buttonVariants };
