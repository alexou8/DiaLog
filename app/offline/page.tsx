import type { Metadata } from 'next';
import { ButtonLink } from '@/components/ui';

export const metadata: Metadata = { title: 'You are offline', robots: { index: false } };

export default function OfflinePage() {
  return (
    <main
      id="main"
      tabIndex={-1}
      className="mx-auto flex min-h-dvh max-w-xl flex-col justify-center px-6 text-center"
    >
      <h1 className="text-3xl font-bold">You&apos;re offline</h1>
      <p className="mt-3 text-ink-muted">
        DiaLog needs a connection to show your readings. Your data is safe. Nothing was lost. This
        page will work again as soon as you&apos;re back online.
      </p>
      {/* A plain link rather than a reload button: the service worker serves
          this page in place of whatever failed, and a link works even if the
          page's scripts never loaded. */}
      <div className="mt-6 flex justify-center">
        <ButtonLink href="/app">Try again</ButtonLink>
      </div>
      <p className="mt-6 text-sm text-ink-muted">
        For your privacy, DiaLog deliberately does not keep your health records on this device.
      </p>
    </main>
  );
}
