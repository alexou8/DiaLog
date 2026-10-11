import * as React from 'react';

import { cn } from '@/lib/utils';

function Input({ className, type, ...props }: React.ComponentProps<'input'>) {
  return (
    <input
      type={type}
      data-slot="input"
      className={cn(
        // 44px floor and full-size text at every breakpoint. Stock shadcn
        // drops to 14px on desktop (`md:text-sm`); health data entry does not
        // get smaller on a bigger screen.
        'min-h-11 w-full min-w-0 rounded-[var(--radius-control)] border-2 border-input bg-surface px-4 py-2.5 text-base text-ink transition-colors selection:bg-primary selection:text-primary-foreground file:inline-flex file:border-0 file:bg-transparent file:text-base file:font-medium file:text-foreground placeholder:text-ink-muted/70 disabled:cursor-not-allowed disabled:opacity-60',
        'focus:border-brand',
        // An invalid field changes shape as well as colour: a thick leading
        // edge, alongside the icon and words in the error under it. Focus
        // still draws the global outline on top, so an invalid field that
        // has focus no longer looks exactly like one that does not.
        'aria-invalid:border-destructive aria-invalid:border-l-[6px]',
        className,
      )}
      {...props}
    />
  );
}

export { Input };
