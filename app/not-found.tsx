import { ButtonLink } from '@/components/ui';

export default function NotFound() {
  return (
    <main className="mx-auto flex min-h-dvh max-w-xl flex-col justify-center px-6 text-center">
      <h1 className="text-3xl font-bold">We couldn&apos;t find that page</h1>
      <p className="mt-3 text-ink-muted">
        The link may be out of date, or the page may have moved. Nothing has happened to your data.
      </p>
      <div className="mt-6 flex justify-center gap-3">
        <ButtonLink href="/">Go to the home page</ButtonLink>
        <ButtonLink href="/app" variant="secondary">
          Open DiaLog
        </ButtonLink>
      </div>
    </main>
  );
}
