'use client';

import { useEffect } from 'react';

/**
 * Last-resort boundary for a failure in the root layout itself, the one place
 * `app/error.tsx` (everything below the root layout) cannot reach. Next replaces the whole
 * document with this component, so it must supply its own <html> and <body>.
 *
 * It is deliberately self-contained: no design-system components, no
 * `globals.css`, no shared helpers. The root layout is exactly what is broken,
 * and anything this imports from the layout's dependency tree could fail the
 * same way. The inline styles below carry their own light and dark palettes
 * (both well above 4.5:1 for text) and respect the larger-text default of the
 * browser by using rem.
 *
 * Like the segment boundaries, it never renders or logs `error.message`: an
 * error raised while handling health data can carry the data. The digest is the
 * only thing logged.
 */
export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error('Global error', error.digest);
  }, [error.digest]);

  return (
    <html lang="en">
      <head>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Something went wrong</title>
        <style>{`
          :root { color-scheme: light dark; }
          body {
            margin: 0;
            min-height: 100vh;
            display: grid;
            place-items: center;
            padding: 1.5rem;
            font-family: ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, Arial, sans-serif;
            font-size: 1.0625rem;
            line-height: 1.6;
            background: #f7f6f2;
            color: #1c2430;
          }
          main { max-width: 34rem; }
          h1 { font-size: 1.5rem; line-height: 1.25; margin: 0 0 0.75rem; }
          p { margin: 0 0 1rem; }
          .actions { display: flex; flex-wrap: wrap; align-items: center; gap: 1rem; }
          button {
            min-height: 2.75rem;
            padding: 0.5rem 1.25rem;
            font: inherit;
            font-weight: 600;
            border-radius: 0.5rem;
            border: 2px solid #155e69;
            background: #155e69;
            color: #ffffff;
            cursor: pointer;
          }
          a { color: #155e69; font-weight: 600; text-underline-offset: 0.25em; }
          :focus-visible { outline: 3px solid #155e69; outline-offset: 3px; }
          @media (prefers-color-scheme: dark) {
            body { background: #14181f; color: #ecf0f4; }
            button { background: #7fd1d9; border-color: #7fd1d9; color: #0b1316; }
            a { color: #7fd1d9; }
            :focus-visible { outline-color: #7fd1d9; }
          }
        `}</style>
      </head>
      <body>
        <main>
          <h1>Something went wrong</h1>
          <p>
            DiaLog could not show this page. Your records are safe: this is a display problem, not a
            data problem. Please try again.
          </p>
          <div className="actions">
            <button type="button" onClick={reset}>
              Try again
            </button>
            {/* A plain anchor, not next/link: the client router may be what failed,
                and a full page load is the reset this boundary exists to offer. */}
            {/* eslint-disable-next-line @next/next/no-html-link-for-pages */}
            <a href="/">Go to the home page</a>
          </div>
        </main>
      </body>
    </html>
  );
}
