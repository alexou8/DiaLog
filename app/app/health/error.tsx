'use client';

import { ErrorState } from '@/components/ui';

export default function HealthError({ reset }: { error: Error & { digest?: string }; reset: () => void }) {
  return (
    <ErrorState title="We could not load your health data">
      <p>Please try again. If this keeps happening, your data is safe — this is just a display issue.</p>
      <button
        type="button"
        onClick={reset}
        className="dl-target mt-3 rounded-xl border-2 border-line-strong px-4 py-2 text-sm font-semibold hover:border-brand"
      >
        Try again
      </button>
    </ErrorState>
  );
}
