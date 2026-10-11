'use client';

import { useEffect } from 'react';
import { ErrorState } from '@/components/ui';

/**
 * Boundary for everything below the root layout that no segment boundary
 * catches: a failure in a section layout (`app/app/layout.tsx`, for example
 * `requireUser` while the database is down) or on a public page. Without it
 * those errors went straight to `app/global-error.tsx`, which replaces the
 * whole document. The root layout is still rendered here, so the design system
 * and the visitor's theme are available; the section layout that failed is
 * not, so this supplies its own `main` landmark.
 *
 * Like every other boundary, it logs only the digest and never shows
 * `error.message`: an error raised while handling health data can carry the
 * data.
 */
export default function RootError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error('Route error', error.digest);
  }, [error.digest]);

  return (
    <main id="main" tabIndex={-1} className="mx-auto max-w-xl px-5 py-12">
      <ErrorState title="Something went wrong">
        <p>
          DiaLog could not show this page. Your records are safe: this is a display problem, not a
          data problem. Please try again.
        </p>
        <div className="mt-3 flex flex-wrap items-center gap-4">
          <button
            type="button"
            onClick={reset}
            className="dl-target rounded-[var(--radius-control)] border-2 border-line-strong px-4 py-2 text-sm font-semibold hover:border-brand"
          >
            Try again
          </button>
          {/* A plain anchor, not next/link: the client router may be what
              failed, and a full page load is the reset this offers. */}
          {/* eslint-disable-next-line @next/next/no-html-link-for-pages */}
          <a href="/" className="dl-target font-semibold underline underline-offset-4">
            Go to the home page
          </a>
        </div>
      </ErrorState>
    </main>
  );
}
