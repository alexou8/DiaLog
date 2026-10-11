import type { Metadata } from 'next';
import { ButtonLink } from '@/components/ui';

// Without this the tab read the site default, so a screen-reader user landing
// on a dead link heard the home page's title rather than that nothing was found.
export const metadata: Metadata = { title: 'Page not found', robots: { index: false } };

export default function NotFound() {
  return (
    <main
      id="main"
      tabIndex={-1}
      className="mx-auto flex min-h-dvh max-w-xl flex-col justify-center px-6 text-center"
    >
      <h1 className="text-3xl font-bold">We couldn&apos;t find that page</h1>
      <p className="mt-3 text-ink-muted">
        The link may be out of date, or the page may have moved. Nothing has happened to your data.
      </p>
      <div className="mt-6 flex flex-wrap justify-center gap-3">
        <ButtonLink href="/">Go to the home page</ButtonLink>
        <ButtonLink href="/app" variant="secondary">
          Open DiaLog
        </ButtonLink>
      </div>
    </main>
  );
}
