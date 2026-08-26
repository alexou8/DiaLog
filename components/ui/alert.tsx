import * as React from 'react';
import { cn } from '@/lib/utils';
import { toneVariants, type Tone } from './tone';

const ALERT_BASE =
  'relative grid w-full grid-cols-[0_1fr] items-start gap-y-1 rounded-[var(--radius-control)] border p-4 text-sm has-[>svg]:grid-cols-[1.15em_1fr] has-[>svg]:gap-x-3 [&>svg]:size-[1.15em] [&>svg]:translate-y-0.5 [&>svg]:text-current';

function Alert({
  className,
  tone = 'info',
  // `role` is overridable because most of these are quiet contextual notes
  // ("note"), not interruptions. Only a genuine failure should be announced
  // assertively as an alert.
  role = 'note',
  ...props
}: React.ComponentProps<'div'> & { tone?: Tone }) {
  return (
    <div
      data-slot="alert"
      role={role}
      className={cn(ALERT_BASE, toneVariants({ tone }), className)}
      {...props}
    />
  );
}

function AlertTitle({ className, ...props }: React.ComponentProps<'div'>) {
  return (
    <div
      data-slot="alert-title"
      className={cn('col-start-2 font-semibold', className)}
      {...props}
    />
  );
}

function AlertDescription({ className, ...props }: React.ComponentProps<'div'>) {
  return (
    <div
      data-slot="alert-description"
      className={cn(
        'dl-measure col-start-2 grid justify-items-start gap-1 text-sm [&_p]:leading-relaxed',
        className,
      )}
      {...props}
    />
  );
}

export { Alert, AlertTitle, AlertDescription };
