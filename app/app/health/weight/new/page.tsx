import type { Metadata } from 'next';
import { requireOnboardedUser } from '@/lib/auth/current-user';
import { Card, PageHeader } from '@/components/ui';
import { WeightForm } from './WeightForm';

export const metadata: Metadata = { title: 'Log weight' };

export default async function NewWeightPage() {
  const user = await requireOnboardedUser();

  return (
    <div className="mx-auto max-w-xl">
      <PageHeader title="Log your weight" />
      <Card>
        <WeightForm timezone={user.profile.timezone} />
      </Card>
    </div>
  );
}
