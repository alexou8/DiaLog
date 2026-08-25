import type { Metadata } from 'next';
import { requireOnboardedUser } from '@/lib/auth/current-user';
import { Card, PageHeader } from '@/components/ui';
import { ActivityForm } from '../ActivityForm';

export const metadata: Metadata = { title: 'Log activity' };

export default async function NewActivityPage() {
  const user = await requireOnboardedUser();

  return (
    <div className="mx-auto max-w-xl">
      <PageHeader title="Log activity" description="What you did, for how long, and how it felt." />
      <Card>
        <ActivityForm timezone={user.profile.timezone} />
      </Card>
    </div>
  );
}
