import type { Metadata } from 'next';
import { requireUser } from '@/lib/auth/current-user';
import { OnboardingForm } from './OnboardingForm';

export const metadata: Metadata = { title: 'Set up DiaLog', robots: { index: false } };

export default async function OnboardingPage() {
  const user = await requireUser();
  return (
    <div className="mx-auto max-w-2xl">
      <h1 className="text-2xl font-semibold tracking-tight sm:text-[1.75rem]">
        Let&apos;s set up DiaLog
      </h1>
      <p className="mt-2 text-ink-muted">
        Five quick questions so the app speaks your language from the start. You can change any of
        this later, and you can skip it entirely.
      </p>
      <OnboardingForm
        defaultName={user.profile.displayName ?? ''}
        defaultTimezone={user.profile.timezone}
      />
    </div>
  );
}
