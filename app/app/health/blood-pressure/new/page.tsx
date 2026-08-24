import type { Metadata } from 'next';
import { requireOnboardedUser } from '@/lib/auth/current-user';
import { Card, PageHeader } from '@/components/ui';
import { BloodPressureForm } from './BloodPressureForm';

export const metadata: Metadata = { title: 'Log blood pressure' };

export default async function NewBloodPressurePage() {
  const user = await requireOnboardedUser();

  return (
    <div className="mx-auto max-w-xl">
      <PageHeader title="Log blood pressure" />
      <Card>
        <BloodPressureForm timezone={user.profile.timezone} />
      </Card>
    </div>
  );
}
