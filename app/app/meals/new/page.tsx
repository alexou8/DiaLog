import type { Metadata } from 'next';
import { requireOnboardedUser } from '@/lib/auth/current-user';
import { Card, PageHeader } from '@/components/ui';
import { MealForm } from '../MealForm';

export const metadata: Metadata = { title: 'Log a meal' };

export default async function NewMealPage() {
  const user = await requireOnboardedUser();

  return (
    <div className="mx-auto max-w-xl">
      <PageHeader
        title="Log a meal"
        description="A quick description and the time are all you need."
      />
      <Card>
        <MealForm timezone={user.profile.timezone} />
      </Card>
    </div>
  );
}
