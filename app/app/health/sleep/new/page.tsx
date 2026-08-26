import type { Metadata } from 'next';
import { requireOnboardedUser } from '@/lib/auth/current-user';
import { Card, PageHeader } from '@/components/ui';
import { SleepForm } from './SleepForm';

export const metadata: Metadata = { title: 'Log sleep' };

export default async function NewSleepPage() {
  const user = await requireOnboardedUser();

  return (
    <div className="mx-auto max-w-xl">
      <PageHeader
        title="Log sleep"
        description="Bedtime and wake time. DiaLog works out the duration."
      />
      <Card>
        <SleepForm timezone={user.profile.timezone} />
      </Card>
    </div>
  );
}
