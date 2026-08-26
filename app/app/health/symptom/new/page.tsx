import type { Metadata } from 'next';
import { requireOnboardedUser } from '@/lib/auth/current-user';
import { toLocalInputValue } from '@/lib/domain/time';
import { Callout, Card, PageHeader } from '@/components/ui';
import { SymptomForm } from './SymptomForm';

export const metadata: Metadata = { title: 'Note how you feel' };

export default async function NewSymptomPage() {
  const { profile } = await requireOnboardedUser();
  return (
    <div className="mx-auto max-w-xl space-y-5">
      <PageHeader
        title="Note how you're feeling"
        description="Recording a symptom next to your readings gives you something concrete to describe at an appointment."
      />
      <Callout tone="info" icon="info" title="This is a note, not an assessment">
        DiaLog records what you write, in your words. It does not interpret symptoms, and it will
        never suggest what they might mean. If something worries you, contact your healthcare
        provider, and in an emergency, your local emergency services.
      </Callout>
      <Card>
        <SymptomForm defaultTime={toLocalInputValue(new Date(), profile.timezone)} />
      </Card>
    </div>
  );
}
