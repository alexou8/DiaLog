import type { Metadata } from 'next';
import { requireOnboardedUser } from '@/lib/auth/current-user';
import { Card, PageHeader } from '@/components/ui';
import { MoodForm } from './MoodForm';

export const metadata: Metadata = { title: 'Log mood' };

export default async function NewMoodPage() {
  const user = await requireOnboardedUser();

  return (
    <div className="mx-auto max-w-xl">
      <PageHeader title="Log your mood" />
      <Card>
        <MoodForm timezone={user.profile.timezone} />
      </Card>
    </div>
  );
}
