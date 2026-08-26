import * as React from 'react';

import { cn } from '@/lib/utils';

function Textarea({ className, ...props }: React.ComponentProps<'textarea'>) {
  return (
    <textarea
      data-slot="textarea"
      className={cn(
        'min-h-28 w-full rounded-[var(--radius-control)] border-2 border-input bg-surface px-4 py-2.5 text-base text-ink outline-none transition-colors field-sizing-content placeholder:text-ink-muted/70 focus:border-brand disabled:cursor-not-allowed disabled:opacity-60 aria-invalid:border-destructive',
        className,
      )}
      {...props}
    />
  );
}

export { Textarea };
