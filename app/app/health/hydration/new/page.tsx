import type { Metadata } from 'next';
import { requireOnboardedUser } from '@/lib/auth/current-user';
import { toLocalInputValue } from '@/lib/domain/time';
import { Card, PageHeader } from '@/components/ui';
import { HydrationForm } from './HydrationForm';

export const metadata: Metadata = { title: 'Log a drink' };

export default async function NewHydrationPage() {
  const { profile } = await requireOnboardedUser();
  return (
    <div className="mx-auto max-w-xl">
      <PageHeader
        title="Log a drink"
        description="Keeping a rough note of what you drink can add useful context to your readings."
      />
      <Card>
        <HydrationForm defaultTime={toLocalInputValue(new Date(), profile.timezone)} />
      </Card>
    </div>
  );
}
