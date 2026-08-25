import type { ReactNode } from 'react';

/** Shared layout for the long-form public pages. */
export function Prose({
  title,
  updated,
  children,
}: {
  title: string;
  updated?: string;
  children: ReactNode;
}) {
  return (
    <article className="mx-auto max-w-2xl px-5 py-12">
      <h1 className="text-3xl font-bold tracking-tight sm:text-4xl">{title}</h1>
      {updated ? <p className="mt-2 text-sm text-ink-muted">Last updated {updated}</p> : null}
      <div className="mt-8 space-y-5 text-lg leading-relaxed [&_h2]:mt-10 [&_h2]:text-xl [&_h2]:font-bold [&_h3]:mt-6 [&_h3]:font-semibold [&_li]:mb-2 [&_ul]:list-disc [&_ul]:pl-6 [&_a]:underline [&_a]:underline-offset-4">
        {children}
      </div>
    </article>
  );
}
