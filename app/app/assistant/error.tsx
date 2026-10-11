'use client';

import { useEffect } from 'react';
import { ErrorState } from '@/components/ui';

export default function AssistantError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    // Only the digest is logged: an opaque id that lets the server-side log be
    // matched to this failure. `error.message` is neither logged nor shown,
    // because an error raised while handling health data can carry the data.
    console.error('Route error', error.digest);
  }, [error.digest]);

  return (
    <ErrorState title="The assistant ran into a problem">
      <p>Your question was not answered. Please try again. Your records are not affected.</p>
      <button
        type="button"
        onClick={reset}
        className="dl-target mt-3 rounded-[var(--radius-control)] border-2 border-line-strong px-4 py-2 text-sm font-semibold hover:border-brand"
      >
        Try again
      </button>
    </ErrorState>
  );
}
