'use client';

import { deleteRecordAction } from '@/lib/actions/records';
import { Button } from '@/components/ui';

/**
 * Two-step delete confirmation built from native <details>/<summary> so it
 * works without JavaScript and is announced correctly by screen readers —
 * no window.confirm().
 */
export function DeleteRecordButton({
  type,
  id,
  label,
}: {
  type: string;
  id: string;
  label: string;
}) {
  return (
    <details className="relative inline-block">
      <summary className="dl-target inline-flex min-h-11 cursor-pointer list-none items-center rounded-xl border-2 border-line-strong px-3 py-2 text-sm font-semibold text-ink-muted marker:content-none hover:border-critical hover:text-critical">
        Delete
      </summary>
      <div className="absolute right-0 z-10 mt-2 w-64 rounded-xl border border-line bg-surface p-3 shadow-lg">
        <p className="mb-3 text-sm">Delete {label}? This cannot be undone.</p>
        <form action={deleteRecordAction} className="flex justify-end gap-2">
          <input type="hidden" name="type" value={type} />
          <input type="hidden" name="id" value={id} />
          <Button type="submit" variant="danger" className="px-3 py-1.5 text-sm">
            Yes, delete
          </Button>
        </form>
      </div>
    </details>
  );
}
