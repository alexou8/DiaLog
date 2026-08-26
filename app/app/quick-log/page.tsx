import type { Metadata } from 'next';
import { requireOnboardedUser } from '@/lib/auth/current-user';
import { toLocalInputValue } from '@/lib/domain/time';
import { ButtonLink, Callout, EmptyState, PageHeader } from '@/components/ui';
import { QuickLogPanel } from './QuickLogPanel';

export const metadata: Metadata = { title: 'Quick logging' };
export const dynamic = 'force-dynamic';

export default async function QuickLogPage() {
  const { profile } = await requireOnboardedUser();

  if (!profile.aiEnabled) {
    return (
      <div className="mx-auto max-w-2xl">
        <PageHeader title="Quick logging" />
        <EmptyState
          title="Quick logging needs the assistant"
          icon="quickLog"
          action={
            <>
              <ButtonLink href="/app/settings">Open settings</ButtonLink>
              <ButtonLink href="/app/meals/new" variant="secondary">
                Use the meal form
              </ButtonLink>
            </>
          }
        >
          <p>
            The assistant is switched off for your account, and quick logging relies on it to turn a
            sentence into entries. The ordinary forms work exactly as usual.
          </p>
        </EmptyState>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-2xl space-y-6">
      <PageHeader
        title="Describe your day"
        description="Type what you ate or did in ordinary words. DiaLog will suggest entries, and nothing is saved until you check them."
      />
      <Callout tone="info" icon="info" title="Every number here is an estimate">
        Nutrition figures worked out from a description are approximations, not measurements. Edit
        anything that looks wrong before you save. You are always the one who decides what gets
        recorded.
      </Callout>
      <QuickLogPanel defaultTime={toLocalInputValue(new Date(), profile.timezone)} />
    </div>
  );
}
