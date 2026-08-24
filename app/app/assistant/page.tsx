import type { Metadata } from 'next';
import { requireOnboardedUser } from '@/lib/auth/current-user';
import { getProvider } from '@/lib/ai/provider';
import { recordCounts } from '@/lib/db/health-records';
import { ButtonLink, Callout, Card, EmptyState, MedicalDisclaimer, PageHeader } from '@/components/ui';
import { AssistantPanel } from './AssistantPanel';

export const metadata: Metadata = { title: 'Assistant' };
export const dynamic = 'force-dynamic';

export default async function AssistantPage() {
  const user = await requireOnboardedUser();
  const counts = await recordCounts(user.id);
  const provider = getProvider(process.env.AI_PROVIDER);
  const hasData = counts.glucose > 0;

  if (!user.profile.aiEnabled) {
    return (
      <div className="mx-auto max-w-2xl">
        <PageHeader title="Assistant" />
        <EmptyState
          title="The assistant is switched off"
          icon="💬"
          action={<ButtonLink href="/app/settings">Open settings</ButtonLink>}
        >
          <p>
            You turned the assistant off for your account. Everything else in DiaLog — your readings,
            charts and observations — keeps working exactly the same.
          </p>
        </EmptyState>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-2xl space-y-6">
      <PageHeader
        title="Ask about your data"
        description="Questions are answered from the observations DiaLog has already worked out from your records — not from your raw readings, and not from anything outside your account."
      />

      <Callout
        tone={provider.isExternal ? 'notice' : 'positive'}
        icon={provider.isExternal ? 'ⓘ' : '🔒'}
        title={provider.isExternal ? `Answers are written by ${provider.name}` : 'Answers are written on this server'}
      >
        {provider.isExternal ? (
          <p>
            Only the summarised findings — figures such as &ldquo;average post-dinner reading across
            14 days&rdquo; — are sent. Your individual readings and your notes are not. You can switch
            this off in Settings.
          </p>
        ) : (
          <p>
            DiaLog is using its built-in explanation engine. Nothing about your health leaves this
            server, and no external AI service is involved.
          </p>
        )}
      </Callout>

      {!hasData ? (
        <EmptyState
          title="There is nothing to ask about yet"
          icon="💬"
          action={<ButtonLink href="/app/glucose/new">Add a reading</ButtonLink>}
        >
          <p>
            The assistant can only answer from your own records. Add a few readings first, and it will
            have something to work with.
          </p>
        </EmptyState>
      ) : (
        <AssistantPanel defaultDetail={user.profile.detailLevel} />
      )}

      <Card>
        <h2 className="font-semibold">What it will not do</h2>
        <p className="mt-2 text-ink-muted">
          It does not diagnose, does not discuss medication doses, and does not tell you to change
          anything about your treatment. If your question needs a clinician, it will say so and help
          you phrase the question for them.
        </p>
        <div className="mt-4">
          <MedicalDisclaimer compact />
        </div>
      </Card>
    </div>
  );
}
