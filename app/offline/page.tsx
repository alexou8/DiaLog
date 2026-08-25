import type { Metadata } from 'next';

export const metadata: Metadata = { title: 'You are offline', robots: { index: false } };

export default function OfflinePage() {
  return (
    <main className="mx-auto flex min-h-dvh max-w-xl flex-col justify-center px-6 text-center">
      <h1 className="text-3xl font-bold">You&apos;re offline</h1>
      <p className="mt-3 text-ink-muted">
        DiaLog needs a connection to show your readings. Your data is safe — nothing was lost. This
        page will work again as soon as you&apos;re back online.
      </p>
      <p className="mt-6 text-sm text-ink-muted">
        For your privacy, DiaLog deliberately does not keep your health records on this device.
      </p>
    </main>
  );
}
